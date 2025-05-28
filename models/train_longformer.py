"""
This is the main script to continue pre-training a Longformer model.
"""
import argparse
import torch
# from transformers.models.deberta_v2.tokenization_deberta_v2_fast import LongformerTokenizer # No longer needed
from torch_xla.runtime import world_size, global_ordinal
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
#import multiprocessing as mp
#import torch.multiprocessing as torchmp
#torchmp.set_sharing_strategy('file_system')

from transformers import (
LongformerTokenizerFast, # Use Longformer tokenizer
LongformerConfig, # Use Longformer config
LongformerForMaskedLM, # Use Longformer model
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

from functools import partial
from itertools import chain

from time import sleep
from lazy_grouping import LazyGroupingDataset

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


def shuffle_and_save_dataset(dataset, output_path, seed=None, shuffle=True):
    # Shuffle only the training dataset
    if shuffle:
        shuffled_train = dataset['train'].shuffle(seed=seed)
    else:
        shuffled_train = dataset['train']

    # Save the shuffled training dataset and the original validation dataset
    shuffled_dataset = DatasetDict({
        'train': shuffled_train,
        'validation': dataset['validation']
    })
    print("Saving shuffled data to disk", flush=True)
    shuffled_dataset.save_to_disk(output_path)
    print(f"Shuffled dataset saved to {output_path}", flush=True)


class ShardedShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, num_shards, shard_id, shuffle_buffer_size, max_steps=None, batch_size=1):
        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_steps = max_steps
        self.batch_size = batch_size

    def __iter__(self):
        buffer = []
        items_yielded = 0
        max_items = self.max_steps * self.batch_size if self.max_steps is not None else float('inf')

        for i, item in enumerate(self.dataset):
            if items_yielded >= max_items:
                break
            if i % self.num_shards == self.shard_id:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    while buffer and items_yielded < max_items:
                        items_yielded += 1
                        yield buffer.pop(0)

        # Yield remaining items
        random.shuffle(buffer)
        while buffer and items_yielded < max_items:
            items_yielded += 1
            yield buffer.pop(0)

def tokenize_function(examples, tokenizer, max_seq_length):
    # here you can actually add a chunker to split the text into smaller parts, of max_len
    return tokenizer(examples["text"],
                    truncation=False,
                    max_length=max_seq_length,
                    padding="max_length")

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


def group_texts(examples, max_seq_length, pad_token=0):
    """
    Group already tokenized texts into chunks of max_seq_length while respecting sample boundaries.

    Args:
        examples: Dictionary with keys like 'input_ids', 'attention_mask', etc. where each value
                 is a list of tokenized examples
        max_seq_length: Maximum sequence length
        pad_token: Token to use for padding (default: 0)

    Returns:
        Dictionary with same keys but values chunked to max_seq_length with padding
    """
    result = {k: [] for k in examples.keys()}

    # Get a sample key to determine the number of examples
    sample_key = list(examples.keys())[0]

    # Loop through each tokenized example
    for i in range(len(examples[sample_key])):
        # Extract the current tokenized example for each feature
        current_example = {k: examples[k][i] for k in examples.keys()}

        # Calculate how many chunks we need for this example
        example_length = len(current_example[sample_key])
        num_chunks = (example_length + max_seq_length - 1) // max_seq_length  # Ceiling division

        # Split each feature into chunks
        for k, tokens in current_example.items():
            # Create chunks of max_seq_length
            chunks = []
            for j in range(0, example_length, max_seq_length):
                chunk = tokens[j:min(j + max_seq_length, example_length)]

                # Pad if necessary
                if len(chunk) < max_seq_length:
                    chunk = chunk + [pad_token] * (max_seq_length - len(chunk))

                chunks.append(chunk)

            # If we don't have enough chunks (unlikely but possible with different length features)
            while len(chunks) < num_chunks:
                chunks.append([pad_token] * max_seq_length)

            # Add the chunks to the result
            result[k].extend(chunks)

    return result

def prep_fn(args):
    # Load and tokenize dataset
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

        # If lazy grouping is enabled, return the raw tokenized dataset
        if args.lazy_grouping:
            print("Lazy grouping enabled. Grouping will be performed on-the-fly during training.", flush=True)
            print(f"This will reduce startup time but may slightly impact training speed.", flush=True)
            return tokenized_dataset_raw

        # Otherwise, perform grouping now as before
        opt_kwargs = {'num_proc': 1, 'desc':f"Grouping texts in chunks of {args.max_seq_length}" } if args.streaming_data==False else {}
        print("Performing chunking tokenized data...", flush=True)
        group_fn = partial(group_texts, max_seq_length=args.max_seq_length, pad_token=args.tokenizer.pad_token_id)
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
            project="Longformer TPU pretraining from scratch",
            config={
                "learning_rate": args.learning_rate,
                "architecture": "Longformer",
                "dataset": args.dataset_dir,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
                "lazy_grouping": args.lazy_grouping
            },
            mode="online",
            dir="/home/bes3/temp"
        )

    # Load model configuration
    #config = RobertaConfig.from_pretrained(args.model_name)

    # Load pre-trained model
    xm.master_print("Loading the LM ...")
    model_config = LongformerConfig.from_pretrained(args.model_name)
    model_config.bos_token_id = args.tokenizer.bos_token_id
    model_config.eos_token_id = args.tokenizer.eos_token_id
    model_config.pad_token_id = args.tokenizer.pad_token_id
    model_config.cls_token_id = args.tokenizer.cls_token_id
    model_config.sep_token_id = args.tokenizer.sep_token_id
    model_config.vocab_size = args.tokenizer.vocab_size
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.hidden_size = args.hidden_size
    model_config.intermediate_size = args.intermediate_size
    model_config.max_position_embeddings = args.max_seq_length

    model = LongformerForMaskedLM(model_config)

    if isinstance(args.checkpoint_path, str) & (args.checkpoint_path != ""):
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
        pass

    model = model.to(device=device, dtype=torch.bfloat16)

    # Set up data collator
    xm.master_print("Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,
                                                    mlm=True,
                                                    mlm_probability=args.mlm_probability)

    # Decide on distributed sampler parameters:
    if args.num_cores == 1:
        sampler_rank = 0
        sampler_replicas = 1
    else:
        sampler_rank = global_ordinal()
        sampler_replicas = world_size()

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

        # Apply lazy grouping if enabled
        if args.lazy_grouping and not args.pre_tokenized:
            print(f"Applying lazy grouping to sharded dataset for worker {shard_id}...", flush=True)
            _sharded_train_dataset = LazyGroupingDataset(
                sharded_train_dataset,
                max_seq_length=args.max_seq_length,
                pad_token=args.tokenizer.pad_token_id,
                batch_size=args.per_device_train_batch_size,
                streaming=args.streaming_data
            )
        else:
            _sharded_train_dataset = sharded_train_dataset

        xm.master_print("Starting the streaming DataLoader...")

        train_dataloader = torch.utils.data.DataLoader(
            _sharded_train_dataset,
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
    save_steps = args.save_epoch_percentage if args.save_epoch_percentage>1. else int(steps_per_epoch * args.save_epoch_percentage)
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
                batch = {k: v.to(device=device, dtype=torch.bfloat16) if v.dtype==torch.float32 else v.to(device) for k, v in batch.items()}
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
                            new_dir_name = f"{args.model_name}_latest"
                            new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                            subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir, "*"), new_gcs_path], check=True)

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
    tokenized_dataset = prep_fn(args)
    train_fn(tokenized_dataset, device, args)

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
    parser.add_argument("--mlm_probability", type=float, default=0.25)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_epoch_percentage", type=float, default=0.05)
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
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument("--lazy_grouping", action='store_true', help="Use lazy grouping to process data on-the-fly during training (incompatible with --pre_tokenized)")
    parser.add_argument("--wandb_key", type=str, required=True,help="Weights & Biases API key")
    args = parser.parse_args()

    #mp.set_start_method('fork', force=True)

    # give error if both keep_in_memory and streaming_data are set to True
    # as they are mutually exclusive
    if args.keep_in_memory and args.streaming_data:
        raise ValueError("keep_in_memory and streaming_data are mutually exclusive. Please set only one of them to True.")

    if args.streaming_data==True and args.shuffle_buffer_size is None:
        raise ValueError("shuffle_buffer_size is required when streaming_data is set to True.")

    if args.lazy_grouping and args.pre_tokenized:
        raise ValueError("lazy_grouping cannot be used with pre_tokenized data. Lazy grouping requires raw tokenized data to group on-the-fly.")

    # Set the same seed for all processes
    torch.manual_seed(args.seed)

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

    args.tokenizer = LongformerTokenizerFast.from_pretrained(args.tokenizer_name_or_path)
    args.tokenizer.model_max_length = args.max_seq_length

    if args.shuffle_dataset:
            args.dataset_dir = args.shuffle_dataset_path
            args.dataset_format = 'arrow'
            print("Continuing with Arrow format", flush=True)

    sleep(20)

    # Spawn processes
    print(f"Initializing TPU...with {args.num_cores} cores")
    print("Spawning processes...")

    if args.num_cores == 1:
        print("Running single process...")
        prep_and_train_fn(0, args)
    else:
        xmp.spawn(prep_and_train_fn, args=(args,), nprocs=args.num_cores )

if __name__ == "__main__":
    print("Starting...")
    main()
