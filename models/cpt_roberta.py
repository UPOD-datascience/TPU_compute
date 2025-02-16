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

from datasets import load_dataset, DatasetDict
import wandb
import os
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
import fsspec

try:
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
except:
    print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

def shuffle_and_save_dataset(dataset, output_path, seed=42):
    # Shuffle only the training dataset
    shuffled_train = dataset['train'].shuffle(seed=seed)

    # Save the shuffled training dataset and the original validation dataset
    shuffled_dataset = DatasetDict({
        'train': shuffled_train,
        'validation': dataset['validation']
    })
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

def prep_fn(args):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        # Split by chunks of max_len.
        # TODO: add sentence/paragraph boundary respecter..
        # TODO: add a chunker that also store the last part of the previous chunk
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Load and tokenize dataset
    if args.pre_tokenized:
        print("Loading pre-tokenized dataset...", flush=True)
        datasets = {
            "train": args.dataset_dir + f"/train_{args.max_seq_length}.{args.dataset_format}",
            "validation": args.dataset_dir + f"/validation_{args.max_seq_length}.{args.dataset_format}"
        }
        if args.streaming_data:
            tokenized_dataset = load_dataset(
                args.dataset_format,
                data_files=datasets,
                streaming=True,
                keep_in_memory=False,
            ).shuffle(buffer_size=args.shuffle_buffer_size)
        else:
            tokenized_dataset = load_dataset(
                args.dataset_format,
                data_files=datasets,
                streaming=False,
                keep_in_memory=True,
            )
    else:
        print(f"Tokenizing dataset... with streaming: {args.streaming_data} and keep_in_memory: {args.keep_in_memory}", flush=True)
        print(f"Dataset location: {args.dataset_dir}", flush=True)

        datasets = {"train": args.dataset_dir+f"/train/*.{args.dataset_format}",
                    "validation": args.dataset_dir+f"/validation/*.{args.dataset_format}"}

        if args.streaming_data:
            dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=True,
                    keep_in_memory=False
                ).shuffle(buffer_size=args.shuffle_buffer_size)
        else:
            dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=False,
                    keep_in_memory=True
                )
        def tokenize_function(examples):
            # here you can actually add a chunker to split the text into smaller parts, of max_len
            return tokenizer(examples["text"],
                             truncation=True,
                             max_length=args.max_seq_length,
                             padding=True) #"max_length")
        opt_kwargs = {'num_proc': args.num_cores} if args.streaming_data==False else {}
        tokenized_dataset_raw = dataset.map(tokenize_function,
                                         batched=True,
                                         remove_columns=["text",
                                                         "id",
                                                         "source",
                                                         "approx_token_counts_translated",
                                                         "approx_token_counts_original"],
                                         **opt_kwargs)

        opt_kwargs = {'num_proc': 1, 'desc':f"Grouping texts in chunks of {args.max_seq_length}" } if args.streaming_data==False else {}
        tokenized_dataset = tokenized_dataset_raw.map(
                group_texts,
                batched=True,
                **opt_kwargs
            )
        del tokenized_dataset_raw
    return tokenized_dataset, tokenizer

def train_fn(index, args):
    # Set up device
    device = xm.xla_device()

    # Initialize wandb for the master process
    if global_ordinal() == 0: #.is_master_ordinal():
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="RoBERTa TPU CPT",
            config={
                "learning_rate": args.learning_rate,
                "architecture": "RoBERTa",
                "dataset": args.dataset_dir,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
            }
        )

    # Load model configuration
    #config = RobertaConfig.from_pretrained(args.model_name)

    # Load pre-trained model
    model = RobertaForMaskedLM.from_pretrained(args.model_name)#, config=config)
    model.to(device)

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,
                                                    mlm=True, mlm_probability=0.15)

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
        num_shards = world_size()  # Total number of TPU cores (or processes)
        shard_id = global_ordinal()  # Unique id for the current process

        sharded_shuffled_dataset = ShardedShuffleDataset(
            args.tokenized_datasets["train"],
            num_shards=num_shards,
            shard_id=shard_id,
            shuffle_buffer_size=args.shuffle_buffer_size,
            max_steps=args.max_steps_per_epoch
        )

        train_dataloader = torch.utils.data.DataLoader(
            sharded_shuffled_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            shuffle=False
        )
    elif args.sharded_data:
        sharded_dataset = args.tokenized_datasets["train"].shard(num_shards=world_size(), index=global_ordinal())

        train_dataloader = torch.utils.data.DataLoader(
            sharded_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True
        )
    else:
        distributed_sampler = True
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            args.tokenized_datasets["train"],
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            shuffle=True
        )
        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            args.tokenized_datasets["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            sampler=train_sampler
        )
    #################
    validation_dataloader = torch.utils.data.DataLoader(
        args.tokenized_datasets["validation"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator
    )

    del args.tokenized_datasets
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
    for epoch in range(args.num_train_epochs):
        total_loss = 0.
        sub_total_loss = 0.
        sub_step = 0
        model.train()

        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(xla_train_loader):
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
                local_avg_loss = total_loss / step
                local_avg_loss_N = sub_total_loss / sub_step
                global_avg_loss = xm.mesh_reduce("loss", local_avg_loss, np.mean)
                global_avg_loss_N = xm.mesh_reduce("loss_N", local_avg_loss_N, np.mean)

                perplexity = math.exp(global_avg_loss)
                perplexity_N = math.exp(global_avg_loss_N)

                xm.master_print(f"Epoch {epoch+1}, step {step}, loss: {global_avg_loss}, perplexity: {perplexity}")

                sub_step = 0
                sub_total_loss = 0.

                if xm.is_master_ordinal():
                    wandb.log({
                        "train_global_average_loss": global_avg_loss,
                        "train_global_average_loss_N": global_avg_loss_N,
                        "train_perplexity": perplexity,
                        "train_perplexity_N": perplexity_N,
                        "epoch": epoch,
                        "step": step,
                        "total_step": total_step
                    })

            total_step += 1
            sub_step +=1


            if (global_ordinal() ==0) & (step % save_steps == 0):
                xm.master_print("Saving model...")
                if args.output_dir.startswith("gs://"):
                    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                        local_output_dir = tmpdirname
                        xm.master_print(f"Saving model to {local_output_dir}...")
                        model_cpu = copy.deepcopy(model).to("cpu")
                        model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                        xm.master_print(f"Uploading model to {args.output_dir}...")
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

        val_ppl = evaluate(model, xla_validation_loader, device)
        xm.master_print(f"Epoch {epoch+1} Validation Perplexity: {val_ppl:.3f}")

        if xm.is_master_ordinal():
            wandb.log({
                "val_perplexity": val_ppl,
                "epoch": epoch,
            })

        # Save model checkpoint
        if global_ordinal() == 0:
            xm.master_print("Saving model...")
            if args.output_dir.startswith("gs://"):
                with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                    local_output_dir = tmpdirname
                    xm.master_print(f"Saving model to {local_output_dir}...")
                    model_cpu = copy.deepcopy(model).to("cpu")
                    model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                    xm.master_print(f"Uploading model to {args.output_dir}...")
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


    # Finish wandb run
    if global_ordinal() == 0:
        wandb.finish()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

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
    parser.add_argument("--dataset_format", default="json", choices=["json", "parquet"])
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
    parser.add_argument("--shuffle_dataset", action='store_true')
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument("--wandb_key", type=str, required=True,help="Weights & Biases API key")
    args = parser.parse_args()

    # give error if both keep_in_memory and streaming_data are set to True
    # as they are mutually exclusive
    if args.keep_in_memory and args.streaming_data:
        raise ValueError("keep_in_memory and streaming_data are mutually exclusive. Please set only one of them to True.")

    if args.streaming_data==True and args.shuffle_buffer_size is None:
        raise ValueError("shuffle_buffer_size is required when streaming_data is set to True.")

    # Set the same seed for all processes
    torch.manual_seed(args.seed)

    if args.shuffle_dataset:
        if xm.is_master_ordinal():
            print("Loading dataset for shuffling...")
            dataset = load_dataset(args.dataset_format, data_files={
                "train": args.dataset_dir + f"/train/*.{args.dataset_format}",
                "validation": args.dataset_dir + f"/validation/*.{args.dataset_format}"
            })
            print("Shuffling and saving dataset...")
            shuffle_and_save_dataset(dataset,
                args.shuffled_dataset_path)

            # Make sure all processes wait for the shuffling to complete
            xm.rendezvous("dataset_shuffled")

            # Update the dataset directory to the shuffled dataset path
            args.dataset_dir = args.shuffled_dataset_path

            # Clear the dataset from memory
            del dataset
            gc.collect()
            print("Cleared shuffled dataset from master's memory")

    tokenized_dataset, tokenizer = prep_fn(args)

    # update args with traindata, valdata, tokenizer
    args.tokenized_datasets = tokenized_dataset
    args.tokenizer = tokenizer

    # Spawn processes
    print(f"Initializing TPU...with {args.num_cores} cores")
    print("Spawning processes...")

    if args.num_cores == 1:
        print("Running single process...")
        train_fn(0, args)
    else:
        xmp.spawn(train_fn, args=(args,))

if __name__ == "__main__":
    print("Starting...")
    main()
