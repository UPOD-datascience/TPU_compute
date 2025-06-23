"""
This is the main script to continue pre-training a Roberta model.
"""
import argparse
import torch
from torch_xla.runtime import world_size, global_ordinal
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import (
RobertaTokenizer,
RobertaConfig,
RobertaForMaskedLM,
DataCollatorForLanguageModeling
)
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup

from datasets import load_dataset, DatasetDict, DatasetInfo
from safetensors.torch import load_file as load_safetensors

import wandb
import os
import io
import sys
import tempfile
import subprocess
import copy
import random
import math
import gc
import datetime
import shutil
import numpy as np
from gcsfs import GCSFileSystem
from google.cloud import storage
import fsspec
from data_loader import DirectoryShardedShuffleDataset, tokenize_and_group, iterate_batches
from functools import partial
from itertools import chain

from time import sleep

try:
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
except:
    print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

print("TPU_NUM_DEVICES:", os.environ.get("TPU_NUM_DEVICES"), flush=True)
print("TPU_CHIPS_PER_HOST_BOUNDS:", os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS"), flush=True)

worker_id = int(os.environ.get('TPU_WORKER_ID', '0'))
# Determine total number of shards (for example, from the comma-separated hostnames)
shards = os.environ.get('TPU_WORKER_HOSTNAMES', '0').split(',')

print(f"TPU_WORKER_ID: {worker_id}")
print(f"TPU_WORKER_HOSTNAMES for {worker_id}: {shards}")

def tokenize_example(example, tokenizer, max_seq_length):
    """
    Helper function to tokenize a single example dictionary.
    Assumes the text is in the 'text' key, or finds the first string field.

    Args:
        example: A dictionary from the dataset (e.g., {'text': '...'}).
        tokenizer: The tokenizer object.
        max_seq_length: Maximum sequence length.

    Returns:
        A dictionary containing tokenized inputs (input_ids, attention_mask).
    """
    if 'text' not in example:
        # Find the first string field if 'text' doesn't exist
        for key, value in example.items():
            if isinstance(value, str):
                text = value
                break
        else:
            # If no string field found, return empty tokenization
            text = ""
            print(f"No text field found in example: {example}")

    else:
        text = example['text']

    # Tokenize the text
    # Add padding=True for batching later with a collator
    # return_tensors='pt' is usually done by the collator, not here
    tokenized = tokenizer(text, truncation=True, max_length=max_seq_length, return_attention_mask=True)

    # DirectoryShardedShuffleDataset yields one item at a time,
    # so we don't group texts here. Grouping happens conceptually
    # via batching and padding. Labels are typically derived in the collator
    # for MLM, or handled by the model depending on the task.

    return tokenized

def shuffle_and_save_dataset(dataset, output_path, seed=None):
    # Shuffle only the training dataset
    shuffled_train = dataset['train'].shuffle(seed=seed)

    # Save the shuffled training dataset and the original validation dataset
    shuffled_dataset = DatasetDict({
        'train': shuffled_train,
        'validation': dataset['validation']
    })
    print("Saving shuffled data to disk", flush=True)
    shuffled_dataset.save_to_disk(output_path)
    print(f"Shuffled dataset saved to {output_path}", flush=True)

def tokenize_function(examples, tokenizer, max_seq_length):
    # here you can actually add a chunker to split the text into smaller parts, of max_len
    return tokenizer(examples["text"],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=True)

def load_from_gcs(bucket_name, blob_name, local_path, device):
    # Initialize a client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the blob to a local file
    try:
        blob.download_to_filename(local_path)
        print(f"Succesfully downloaded checkpoint to: {local_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download checkpoint to {local_path}: {e}")

    # Load the checkpoint based on the file extension
    if local_path.endswith('.safetensors'):
        checkpoint = load_safetensors(local_path, device='cpu')
        checkpoint = {k: v.to(xm.xla_device()) for k, v in checkpoint.items()}
    else:
        checkpoint = torch.load(local_path, map_location=device)

    return checkpoint

def load_sharded_dataset(datasets, dformat):
    num_shards = max(world_size() , 16)
    shard_idx = global_ordinal()

    print(f"Sharding for shard index:{shard_idx} / {num_shards}")

    dataset_dict = load_dataset(
        dformat,
        data_files=datasets,
        streaming=False,
        keep_in_memory=True
    )

    # Apply sharding to each split
    sharded_dataset = {}
    for split in dataset_dict.keys():
        sharded_dataset[split] = dataset_dict[split].shard(num_shards=num_shards, index=shard_idx)

    del dataset_dict
    gc.collect()
    # Combine sharded splits back into a DatasetDict
    return DatasetDict(sharded_dataset)


def group_texts(examples, max_seq_length):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    # TODO: add sentence/paragraph boundary respecter..
    # TODO: add a chunker that also store the last part of the previous chunk
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

def prep_fn(args):
    # Load and tokenize dataset
    #
    if args.use_directory_sharded_dataloader and args.train_data_path:
        print(f"[Process {global_ordinal()}] Using DirectoryShardedShuffleDataset", flush=True)

        # Get GCS project ID from environment if available
        gcs_project = os.environ.get("PROJECT_ID", None)

        # Create the training dataset
        train_dataset_iter = DirectoryShardedShuffleDataset(
            data_path=args.train_data_path,
            buffer_size=args.shuffle_buffer_size,
            file_extension=".json",  # Assuming .jsonl is your file extension
            seed=args.seed + global_ordinal(),   # Add process index to seed for diversity
            gcs_project=gcs_project,
            shuffle_files=True,       # Shuffle file order for training
            shuffle_lines=True        # Shuffle lines within buffer for training
        )

        # Create validation dataset if path exists
        val_dataset_iter = None
        if args.val_data_path:
            val_dataset_iter = DirectoryShardedShuffleDataset(
                data_path=args.val_data_path,
                buffer_size=args.shuffle_buffer_size // 2,  # Smaller buffer for validation
                file_extension=".json",
                seed=args.seed,       # Fixed seed for reproducible validation
                gcs_project=gcs_project,
                shuffle_files=False,  # Generally don't shuffle files for validation
                shuffle_lines=False   # Generally don't shuffle lines for validation
            )

        # Process the datasets (tokenization, etc.)
        # Wrap datasets in iterables that apply transformations
        tokenizer = args.tokenizer
        train_iter = map(lambda example: tokenize_and_group(example, tokenizer, args.max_seq_length), train_dataset_iter)
        val_iter = None if val_dataset_iter is None else map(
            lambda example: tokenize_and_group(example, tokenizer, args.max_seq_length),
            val_dataset_iter
        )

        # The train_fn function will need to handle these iterables directly
        # rather than expecting Hugging Face Dataset objects
        return train_iter, val_iter
    else:
        if args.pre_tokenized:
            datasets = {
                "train": args.dataset_dir + f"/train_{args.max_seq_length}.{args.dataset_format}",
                "validation": args.dataset_dir + f"/validation_{args.max_seq_length}.{args.dataset_format}"
            }
            if args.streaming_data:
                print("Loading pre-tokenized dataset, streaming with shuffle..", flush=True)
                tokenized_dataset = load_dataset(
                        args.dataset_format,
                        data_files=datasets,
                        streaming=True,
                        keep_in_memory=False,
                    ).shuffle(buffer_size=args.shuffle_buffer_size)
            else:
                print("Loading pre-tokenized dataset, no streaming", flush=True)
                tokenized_dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=False,
                    keep_in_memory=True,
                )
        else:
            print(f"Tokenizing dataset... with streaming: {args.streaming_data} and keep_in_memory: {args.keep_in_memory}", flush=True)
            print(f"Dataset location: {args.dataset_dir}", flush=True)

            train_loc = "train" if not args.debug else "validation"

            datasets = {"train": args.dataset_dir+f"/{train_loc}/*.{args.dataset_format}",
                        "validation": args.dataset_dir+f"/validation/*.{args.dataset_format}"}

            if args.streaming_data:
                try:
                    ds_train_info = DatasetInfo.from_directory(args.dataset_dir+f"/{train_loc}/")
                    num_examples = ds_train_info['num_examples']
                    args.max_steps_per_epoch = num_examples // args.per_device_train_batch_size // max(world_size(), 1)
                except Exception as e:
                    print(f"Could not obtain datasetinfo:{e}")
                    pass

                print("Init streaming dataset...", flush=True)
                dataset = load_dataset(
                        args.dataset_format,
                        data_files=datasets,
                        streaming=True,
                        keep_in_memory=False
                    )#.shuffle(buffer_size=args.shuffle_buffer_size)
            else:
                print("Init non-streaming dataset...", flush=True)
                if args.sharded_data:
                    print("Sharding data...", flush=True)
                    dataset = load_sharded_dataset(datasets, args.dataset_format)
                else:
                    dataset = load_dataset(
                            args.dataset_format,
                            data_files=datasets,
                            streaming=False,
                            keep_in_memory=True
                        )

            opt_kwargs = {'num_proc': args.num_cores} if args.streaming_data==False else {}
            print("Tokenizing dataset...", flush=True)
            tokenize_fn = partial(tokenize_function, tokenizer=args.tokenizer, max_seq_length=args.max_seq_length)
            tokenized_dataset_raw = dataset.map(tokenize_fn,
                                            batched=True,
                                            remove_columns=["text",
                                                            "id",
                                                            "source",
                                                            "approx_token_counts_translated",
                                                            "approx_token_counts_original"],
                                            **opt_kwargs)

            opt_kwargs = {'num_proc': 1, 'desc':f"Grouping texts in chunks of {args.max_seq_length}" } if args.streaming_data==False else {}
            print("Performing chunking tokenized data...", flush=True)
            group_fn = partial(group_texts, max_seq_length=args.max_seq_length)
            tokenized_dataset = tokenized_dataset_raw.map(
                    group_fn,
                    batched=True,
                    **opt_kwargs
                )
            del tokenized_dataset_raw

        return tokenized_dataset

def safe_iter(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except Exception as e:
            print(f"Error encountered in iteration: {e}")
            continue

def train_fn(tokenized_dataset, device, args):
    # Initialize wandb for the master process
    if global_ordinal() == 0: #.is_master_ordinal():
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="RoBERTa TPU CPT",
            config={
                "TPU ID": args.TPU_NAME,
                "TPU DISK": args.TPU_DISK,
                "learning_rate": args.learning_rate,
                "architecture": "RoBERTa",
                "dataset": args.dataset_dir,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
            },
            mode="online",
            dir="/home/bes3/temp"
        )

    # Load model configuration
    #config = RobertaConfig.from_pretrained(args.model_name)

    # Load pre-trained model
    xm.master_print("Loading the LM ...")
    if isinstance(args.checkpoint_path, str) & (args.checkpoint_path != ""):
        model = RobertaForMaskedLM.from_pretrained(args.model_name)
        if args.checkpoint_path.startswith('gs://'):
            # Parse GCS path
            bucket_name = args.checkpoint_path.split('/')[2]
            blob_name = '/'.join(args.checkpoint_path.split('/')[3:])
            local_path = f'/tmp/checkpoint.{args.checkpoint_path.split(".")[-1]}'  # Temporary local path to store the downloaded file
            checkpoint = load_from_gcs(bucket_name,
                                       blob_name,
                                       local_path,
                                       device)
        else:
            if args.checkpoint_path.endswith('.safetensors'):
                checkpoint = load_safetensors(args.checkpoint_path, device=device)
            else:
                checkpoint = torch.load(args.checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = RobertaForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    # Set up data collator
    xm.master_print("Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,
                                                    mlm=True, mlm_probability=args.mlm_probability)

    # Decide on distributed sampler parameters:
    if args.num_cores == 1:
        sampler_rank = 0
        sampler_replicas = 1
    else:
        sampler_rank = global_ordinal()
        sampler_replicas = world_size()

    using_directory_sharded = args.use_directory_sharded_dataloader and \
                              args.train_data_path and \
                              isinstance(tokenized_dataset, tuple)

    if using_directory_sharded:
        distributed_sampler = False
        train_data_ds = tokenized_dataset[0]
        validation_data_ds = tokenized_dataset[1]

        train_dataset_mapped = map(
            lambda example: tokenize_example(example, args.tokenizer, args.max_seq_length),
            train_data_ds
        )
        validation_dataset_mapped = map(
            lambda example: tokenize_example(example, args.tokenizer, args.max_seq_length),
            validation_data_ds
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_mapped,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0, # Multiprocessing is handled by xmp.spawn
            pin_memory=False # Not typically needed for XLA
        )

        validation_dataloader = None
        if validation_dataset_mapped is not None:
            validation_dataloader = torch.utils.data.DataLoader(
                validation_dataset_mapped,
                batch_size=args.per_device_train_batch_size, # Use same batch size for eval
                collate_fn=data_collator,
                num_workers=0,
                pin_memory=False
            )
    else:
        # Create sampler
        distributed_sampler = False
        if args.streaming_data:
            xm.master_print("Instantiate sharded dataset...")
            num_shards = world_size()  # Total number of TPU cores (or processes)
            shard_id = global_ordinal()  # Unique id for the current process
            print(f"Num shards: {num_shards}, shard id: {shard_id}")
            #num_shards = xm.xrt_world_size()
            #shard_id = xm.get_ordinal()
            sharded_train_dataset = tokenized_dataset["train"].shard(num_shards=num_shards, index=shard_id)

            xm.master_print("Starting the streaming DataLoader...")

            train_dataloader = torch.utils.data.DataLoader(
                sharded_train_dataset,
                batch_size=args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=0
            )
        else:
            xm.master_print("Starting the DistributedSampler...")
            distributed_sampler = True
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                tokenized_dataset["train"],
                num_replicas=sampler_replicas,
                rank=sampler_rank,
                shuffle=True
            )
            # Create dataloaders
            xm.master_print("Starting the DataLoader...")
            train_dataloader = torch.utils.data.DataLoader(
                tokenized_dataset["train"],
                batch_size=args.per_device_train_batch_size,
                collate_fn=data_collator,
                sampler=train_sampler
            )
        #################
        validation_dataloader = torch.utils.data.DataLoader(
            tokenized_dataset["validation"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator)

        if not args.streaming_data:
            del tokenized_dataset
            gc.collect()

    xla_train_loader = pl.MpDeviceLoader(train_dataloader, xm.xla_device())
    xla_validation_loader = pl.MpDeviceLoader(validation_dataloader, xm.xla_device())

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_steps_per_epoch if args.streaming_data else len(train_dataloader)
    save_steps = int(steps_per_epoch * args.save_epoch_percentage)
    total_steps = steps_per_epoch * args.num_train_epochs
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)

    if global_ordinal() == 0: # xm.is_master_ordinal():
        print("Starting training...", flush=True)
        print(f"Total steps: {total_steps}", flush=True)
        print(f"Total epochs: {args.num_train_epochs}", flush=True)
        print(f"Total warmup steps: {args.num_warmup_steps}", flush=True)

    # Training loop
    total_step = 0
    miss_steps = 0
    print(f"ENTERING THE TRAINING LOOP with: {args.num_train_epochs} epochs", flush=True)
    for epoch in range(args.num_train_epochs):
        print(f"Starting with epoch {epoch}...", flush=True)
        total_loss = 0.
        sub_total_loss = 0.
        sub_step = 0
        model.train()
        if distributed_sampler:
            print(f"Starting with epoch {epoch}...", flush=True)
            train_sampler.set_epoch(epoch)
            print("done with setting epoch..", flush=True)

        print(f"Entering data loader iterator for epoch {epoch}...", flush=True)
        step = -1
        for step, batch in enumerate(safe_iter(xla_train_loader)):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                total_loss += loss.item()
                sub_total_loss += loss.item()

                if  (step+1) % args.gradient_accumulation_steps == 0:
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    optimizer.zero_grad()

                    total_step += args.gradient_accumulation_steps

                if (step+1) % args.logging_steps == 0: # xm.is_master_ordinal():
                    xm.rendezvous("logging")

                    local_avg_loss = total_loss / step
                    local_avg_loss_N = sub_total_loss / sub_step
                    global_avg_loss = xm.mesh_reduce("loss", local_avg_loss, np.mean)
                    global_avg_loss_N = xm.mesh_reduce("loss_N", local_avg_loss_N, np.mean)

                    perplexity = math.exp(global_avg_loss)
                    perplexity_N = math.exp(global_avg_loss_N)

                    sub_step = 0
                    sub_total_loss = 0.

                    if global_ordinal() ==0:
                        print(f"Logging for epoch: {epoch}, step: {step}, train_perplexity_N:{perplexity_N}", flush=True)
                        wandb.log({
                            "train_global_average_loss": global_avg_loss,
                            "train_global_average_loss_N": global_avg_loss_N,
                            "train_perplexity": perplexity,
                            "train_perplexity_N": perplexity_N,
                            "epoch": epoch,
                            "step": step,
                            "total_step": total_step
                        })

                if args.debug:
                    break

                total_step += 1
                sub_step +=1


                if (global_ordinal() ==0) & ((step+1) % save_steps == 0):
                    xm.rendezvous("before_save")
                    print("Saving model...", flush=True)
                    if args.output_dir.startswith("gs://"):
                        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                            local_output_dir = tmpdirname
                            print(f"Saving model to {local_output_dir}...", flush=True)
                            model_cpu = copy.deepcopy(model).to("cpu")
                            model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                            print(f"Uploading model to {args.output_dir}...", flush=True)
                            subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
                            # rename the directory on GCS to the model name, date, and epoch_step
                            # Generate a new directory name with model name, date, and epoch_step
                            current_time = datetime.datetime.now().strftime("%Y%m%d")
                            new_dir_name = f"{args.model_name}_epoch{epoch}_step{step}_{current_time}"
                            new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                            # Rename the directory on GCS
                            subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir), new_gcs_path], check=True)

                    else:
                        model_cpu = copy.deepcopy(model).to("cpu")
                        model_cpu.save_pretrained(args.output_dir, save_serialization=True)
                    xm.rendezvous("after_save")
                else:
                    xm.rendezvous("before_save")
                    xm.rendezvous("after_save")

            except Exception as e:
                print(f"Error occurred during processing batch {step}: {e}", flush=True)
                miss_steps += 1
                continue

            if (step+1) % steps_per_epoch == 0:
                break

        val_ppl = evaluate(model, xla_validation_loader, device)

        if global_ordinal() == 0:
            wandb.log({
                "val_perplexity": val_ppl,
                "epoch": epoch,
                "miss_steps": miss_steps
            })

        xm.rendezvous("syncing for next epoch.")

        # Save model checkpoint
        if global_ordinal() == 0:
            xm.rendezvous("before_epoch_save")
            print("Saving model...", flush=True)
            if args.output_dir.startswith("gs://"):
                with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                    local_output_dir = tmpdirname
                    print(f"Saving model to {local_output_dir}...", flush=True)
                    model_cpu = copy.deepcopy(model).to("cpu")
                    model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                    print(f"Uploading model to {args.output_dir}...", flush=True)
                    subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
                    # rename the directory on GCS to the model name, date, and epoch_step
                    # Generate a new directory name with model name, date, and epoch_step
                    current_time = datetime.datetime.now().strftime("%Y%m%d")
                    new_dir_name = f"{args.model_name}_epoch{epoch}_{current_time}"
                    new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                    # Rename the directory on GCS
                    subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir), new_gcs_path], check=True)
            else:
                model_cpu = copy.deepcopy(model).to("cpu")
                model_cpu.save_pretrained(args.output_dir, save_serialization=True)
            xm.rendezvous("after_epoch_save")
        else:
            xm.rendezvous("before_epoch_save")
            xm.rendezvous("after_epoch_save")

    # Finish wandb run
    if global_ordinal() == 0:
        wandb.finish()

def prep_and_train_fn(index, args):
    device = xm.xla_device()
    tokenized_datasets = prep_fn(args)
    train_fn(tokenized_datasets, device, args)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    # Synchronize all TPU cores before starting evaluation
    xm.rendezvous("evaluation")

    # Loop over the validation set
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        loss = outputs.loss

        # Accumulate the loss locally (on this TPU core)
        total_loss += loss.item()
        total_steps += 1

    # Average loss for this core
    local_avg_loss = total_loss / total_steps

    # Now reduce across all TPU cores to get a "global" average
    # mesh_reduce can apply a function (here `np.mean`) over the local values
    global_avg_loss = xm.mesh_reduce("eval_loss", local_avg_loss, np.mean)

    # Perplexity = exp(average cross-entropy loss)
    ppl = math.exp(global_avg_loss)

    model.train()  # Switch back to train mode
    return ppl

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_format", default="json", choices=["json", "parquet", "arrow"])
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pre_tokenized", action='store_true', default=False)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--mlm_probability", type=float, default=0.25)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_epoch_percentage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--keep_in_memory", action='store_true')
    parser.add_argument("--streaming_data", action='store_true')
    parser.add_argument("--sharded_data", action='store_true')
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--shuffle_dataset_path", type=str, default="/home/bob/tmp/shuffle.parquet")
    parser.add_argument("--shuffle_dataset_ext", type=str, default=None)
    parser.add_argument("--shuffle_dataset", action='store_true')
    parser.add_argument("--shuffle_force_update", action='store_true')
    parser.add_argument("--use_directory_sharded_dataloader", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument("--wandb_key", type=str, required=True,help="Weights & Biases API key")
    parser.add_argument("--TPU_NAME", type=str, default="unknown, please set with --TPU_ID")
    parser.add_argument("--TPU_DISK", type=str, default="unknown, please set with --TPU_DISK")
    args = parser.parse_args()

    # give error if both keep_in_memory and streaming_data are set to True
    # as they are mutually exclusive
    if args.keep_in_memory and args.streaming_data:
        raise ValueError("keep_in_memory and streaming_data are mutually exclusive. Please set only one of them to True.")

    if args.streaming_data==True and args.shuffle_buffer_size is None:
        raise ValueError("shuffle_buffer_size is required when streaming_data is set to True.")

    # Set the same seed for all processes
    torch.manual_seed(args.seed)

    if args.use_directory_sharded_dataloader == False:
        if args.shuffle_dataset:
            shuffle_dir = args.shuffle_dataset_path
            if (os.path.exists(shuffle_dir)) and (args.shuffle_force_update==False):
                print(f"Shuffled dataset exists, skipping creation: {shuffle_dir}", flush=True)
            else:
                if (os.path.exists(shuffle_dir)):
                    print(f"Removing shuffled dataset: {shuffle_dir}", flush=True)
                    shutil.rmtree(shuffle_dir)

                if args.shuffle_dataset_ext is not None:
                    print("Loading pre-shuffled dataset...", flush=True)
                    dataset = load_dataset(args.dataset_format, data_files={
                        "train": args.shuffle_dataset_ext,
                        "validation": args.dataset_dir + f"/validation/*.{args.dataset_format}"
                    }, keep_in_memory=True)

                    print("Clearing cache!", flush=True)
                    cache_dir = os.path.expanduser("~/.cache/huggingface")
                    if os.path.exists(cache_dir):
                        print(f"Removing Hugging Face cache directory: {cache_dir}", flush=True)
                        shutil.rmtree(cache_dir)
                    else:
                        print("Hugging Face cache directory not found.", flush=True)

                    dataset.save_to_disk(args.shuffle_dataset_path)

                    print("Clearing cache a-posteriori !", flush=True)
                    cache_dir = os.path.expanduser("~/.cache/huggingface")
                    if os.path.exists(cache_dir):
                        print(f"Removing Hugging Face cache directory: {cache_dir}", flush=True)
                        shutil.rmtree(cache_dir)
                    else:
                        print("Hugging Face cache directory not found.", flush=True)

                else:
                    print("Loading dataset for shuffling...", flush=True)
                    dataset = load_dataset(args.dataset_format, data_files={
                        "train": args.dataset_dir + f"/train/*.{args.dataset_format}",
                        "validation": args.dataset_dir + f"/validation/*.{args.dataset_format}"
                    }, keep_in_memory=True)

                    print("Clearing cache!", flush=True)
                    cache_dir = os.path.expanduser("~/.cache/huggingface")
                    if os.path.exists(cache_dir):
                        print(f"Removing Hugging Face cache directory: {cache_dir}", flush=True)
                        shutil.rmtree(cache_dir)
                    else:
                        print("Hugging Face cache directory not found.", flush=True)

                    print("Shuffling and saving dataset...", flush=True)
                    shuffle_and_save_dataset(dataset, args.shuffle_dataset_path)

                    print("Clearing cache a-posteriori !", flush=True)
                    cache_dir = os.path.expanduser("~/.cache/huggingface")
                    if os.path.exists(cache_dir):
                        print(f"Removing Hugging Face cache directory: {cache_dir}", flush=True)
                        shutil.rmtree(cache_dir)
                    else:
                        print("Hugging Face cache directory not found.", flush=True)

                # Update the dataset directory to the shuffled dataset path
                # Clear the dataset from memory
                del dataset
                gc.collect()
                print("Cleared shuffled dataset from master's memory", flush=True)
        else:
            shuffle_dir = args.shuffle_dataset_path
            if os.path.exists(shuffle_dir):
                print(f"Removing shuffled dataset: {shuffle_dir}", flush=True)
                shutil.rmtree(shuffle_dir)

        args.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

        if args.shuffle_dataset:
                args.dataset_dir = args.shuffle_dataset_path
                args.dataset_format = 'arrow'
                print("Continuing with Arrow format", flush=True)

        sleep(20)
    else:
        # use_directory_sharded_dataloader
        print("Continuing with directory-sharded dataloader..", flush=True)
        args.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)
        # Get the environment variables for the data buckets
        train_data_path = os.environ.get("DATA_BUCKET_TRAIN_NORMALISED")
        val_data_path = os.environ.get("DATA_BUCKET_VAL_NORMALISED")

        if not train_data_path:
            print("WARNING: DATA_BUCKET_TRAIN_NORMALISED environment variable not set.", flush=True)
            print("Using default dataset directory from args.dataset_dir instead.", flush=True)
            train_data_path = os.path.join(args.dataset_dir, "train")

        if not val_data_path:
            print("WARNING: DATA_BUCKET_VAL_NORMALISED environment variable not set.", flush=True)
            print("Using default validation directory from args.dataset_dir instead.", flush=True)
            val_data_path = os.path.join(args.dataset_dir, "validation")

        print(f"Using DirectoryShardedShuffleDataset with the following paths:", flush=True)
        print(f"  Training data: {train_data_path}", flush=True)
        print(f"  Validation data: {val_data_path}", flush=True)

        # Store these paths in args for use in prep_fn
        args.train_data_path = train_data_path
        args.val_data_path = val_data_path

    # Spawn processes
    print(f"Initializing TPU...with {args.num_cores} cores")
    print("Spawning processes...")

    if args.num_cores == 1:
        print("Running single process...")
        prep_and_train_fn(0, args)
    else:
        xmp.spawn(prep_and_train_fn, args=(args,))

if __name__ == "__main__":
    print("Starting...")
    main()
