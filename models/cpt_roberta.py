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

from datasets import load_dataset
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

try:
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
except:
    print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

# class ShardedShuffleDataset(IterableDataset):
#     def __init__(self, dataset, num_shards, shard_id, shuffle_buffer_size):
#         self.dataset = dataset
#         self.num_shards = num_shards
#         self.shard_id = shard_id
#         self.shuffle_buffer_size = shuffle_buffer_size

#     def __iter__(self):
#         sharded_data = itertools.islice(self.dataset, self.shard_id, None, self.num_shards)

#         buffer = []
#         for item in itertools.cycle(sharded_data):  # Use cycle to repeat the dataset
#             buffer.append(item)
#             if len(buffer) >= self.shuffle_buffer_size:
#                 random.shuffle(buffer)
#                 while len(buffer) > self.shuffle_buffer_size // 2:
#                     yield buffer.pop(0)

class ShardedShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, num_shards, shard_id, shuffle_buffer_size):
        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.shard_id
            iter_end = None
        else:
            per_worker = int(math.ceil((self.num_shards - self.shard_id) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.shard_id + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)

        buffer = []
        for i, item in enumerate(self.dataset):
            if i % self.num_shards == self.shard_id:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    while len(buffer) > self.shuffle_buffer_size // 2:
                        yield buffer.pop(0)

        # Yield remaining items
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop(0)

def prep_fn(args):
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Load and tokenize dataset
    if args.pre_tokenized:
        datasets = {"train": args.dataset_dir+f"/train_{args.max_seq_length}.json",
                    "validation": args.dataset_dir+f"/validation_{args.max_seq_length}.json"}
        tokenized_dataset = load_dataset("json", data_files=datasets, streaming=args.streaming_data, keep_in_memory=args.keep_in_memory)
    else:
        datasets = {"train": args.dataset_dir+f"/train/*.json",
                    "validation": args.dataset_dir+f"/validation/*.json"}    
        dataset = load_dataset(args.dataset_dir, streaming=args.streaming_data, keep_in_memory=args.keep_in_memory)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length, padding="max_length")

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

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
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer, mlm=True, mlm_probability=0.15)

    # Decide on distributed sampler parameters:
    if args.num_cores == 1:
        sampler_rank = 0
        sampler_replicas = 1
    else:
        sampler_rank = global_ordinal()
        sampler_replicas = world_size()

    # Create sampler
    if args.streaming_data:
        num_shards = world_size()  # Total number of TPU cores (or processes)
        shard_id = global_ordinal()  # Unique id for the current process

        sharded_shuffled_dataset = ShardedShuffleDataset(
            args.tokenized_datasets["train"],
            num_shards=num_shards,
            shard_id=shard_id,
            shuffle_buffer_size=args.shuffle_buffer_size
        )

        train_dataloader = torch.utils.data.DataLoader(
            sharded_shuffled_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0
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
        model.train()
        for step, batch in enumerate(xla_train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            if  (step+1) % args.gradient_accumulation_steps == 0:
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()

                total_loss +=loss.item()
                total_step += args.gradient_accumulation_steps

            if (step % args.logging_steps == 0): # xm.is_master_ordinal():
                local_avg_loss = total_loss / args.logging_steps
                global_avg_loss = xm.mesh_reduce("loss", local_avg_loss, np.mean)
                perplexity = math.exp(global_avg_loss)
                xm.master_print(f"Epoch {epoch+1}, step {step}, loss: {global_avg_loss}")

                total_loss = 0.
                if xm.is_master_ordinal():
                    wandb.log({
                        "train_global_average_loss": global_avg_loss,
                        "train_perplexity": perplexity,
                        "epoch": epoch,
                        "step": step,
                        "total_step": total_step
                    })

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
