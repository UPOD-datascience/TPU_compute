"""
This is the main script to continue pre-training a Roberta model on GPU.
"""
import argparse
import torch

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
from itertools import chain
from functools import partial


def prep_fn(args):
    def group_texts(examples, pad_token=0):
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
            num_chunks = (example_length + args.max_seq_length - 1) // args.max_seq_length  # Ceiling division

            # Split each feature into chunks
            for k, tokens in current_example.items():
                # Create chunks of max_seq_length
                chunks = []
                for j in range(0, example_length, args.max_seq_length):
                    chunk = tokens[j:min(j + args.max_seq_length, example_length)]

                    # Pad if necessary
                    if len(chunk) < args.max_seq_length:
                        chunk = chunk + [pad_token] * (args.max_seq_length - len(chunk))

                    chunks.append(chunk)

                # If we don't have enough chunks (unlikely but possible with different length features)
                while len(chunks) < num_chunks:
                    chunks.append([pad_token] * args.max_seq_length)

                # Add the chunks to the result
                result[k].extend(chunks)

        return result

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Load and tokenize dataset
    if args.pre_tokenized:
        print("Loading pre-tokenized dataset...", flush=True)
        datasets = {
            "train": args.dataset_dir + f"/train_{args.max_seq_length}.json",
            "validation": args.dataset_dir + f"/validation_{args.max_seq_length}.json"
        }
        tokenized_dataset = load_dataset(
            "json",
            data_files=datasets,
            streaming=args.streaming_data,
            keep_in_memory=args.keep_in_memory
        )
    else:
        print(f"Tokenizing dataset... with streaming: {args.streaming_data} and keep_in_memory: {args.keep_in_memory}", flush=True)
        datasets = {"train": args.dataset_dir+f"/train/*.json",
                    "validation": args.dataset_dir+f"/validation/*.json"}
        dataset = load_dataset(args.dataset_dir, streaming=args.streaming_data, keep_in_memory=args.keep_in_memory)

        def tokenize_function(examples):
            # here you can actually add a chunker to split the text into smaller parts, of max_len
            return tokenizer(examples["text"],
                             truncation=False,
                             max_length=args.max_seq_length)
        opt_kwargs = {'num_proc': 8} if args.streaming_data==False else {}

        tokenized_dataset_raw = dataset.map(tokenize_function,
                                         batched=True,
                                         remove_columns=["text",
                                                         "id",
                                                         "source",
                                                         "approx_token_counts_translated",
                                                         "approx_token_counts_original"],
                                         **opt_kwargs)

        opt_kwargs = {'num_proc': 1, 'desc':f"Grouping texts in chunks of {args.max_seq_length}" } if args.streaming_data==False else {}
        group_fn = partial(group_texts, pad_token=tokenizer.pad_token_id)
        tokenized_dataset = tokenized_dataset_raw.map(
                group_fn,
                batched=True,
                **opt_kwargs
            )
        del tokenized_dataset_raw
    return tokenized_dataset, tokenizer

def get_optimizer(model, args):
    """
    Initializes AdamW optimizer with warm-up and linear decay.
    Excludes weight decay for biases and LayerNorm weights.
    """

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return optimizer

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

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
        total_loss += loss.item()
        total_steps += 1

    global_avg_loss = total_loss / total_steps
    ppl = math.exp(global_avg_loss)
    model.train()  # Switch back to train mode
    return ppl, global_avg_loss

def train_fn(index, args):
    # Set up device for GPU (or CPU if no GPU available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    print(50*"+")

    # Initialize wandb for the master process (we use index==0 as master)
    if index == 0:
        wandb.init(
            project="RoBERTa GPU CPT",
            config={
                "learning_rate": args.learning_rate,
                "architecture": "RoBERTa",
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
            }
        )

    # Load pre-trained model
    model = RobertaForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,
                                                     mlm=True,
                                                     mlm_probability=0.15)

    # For GPU training, assume a single process (non-distributed)
    sampler_rank = 0
    sampler_replicas = 1

    # Create dataloader
    if args.streaming_data:
        # For GPU, simply use the dataset without sharding
        print("Loading streaming dataloader..")
        train_dataloader = torch.utils.data.DataLoader(
            args.tokenized_datasets["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0
        )
    elif args.sharded_data:
        print("Loading sharded dataloader")
        sharded_dataset = args.tokenized_datasets["train"].shard(num_shards=1, index=0)
        train_dataloader = torch.utils.data.DataLoader(
            sharded_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True
        )
    else:
        print("Loading normal dataloader")
        train_dataloader = torch.utils.data.DataLoader(
            args.tokenized_datasets["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True
        )

    validation_dataloader = torch.utils.data.DataLoader(
        args.tokenized_datasets["validation"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator
    )

    del args.tokenized_datasets
    gc.collect()

    # For GPU training, use the dataloaders directly (no device loader wrapper needed)
    train_loader = train_dataloader
    validation_loader = validation_dataloader

    # Set up optimizer and scheduler
    optimizer = get_optimizer(model, args)
    steps_per_epoch = args.max_steps_per_epoch if args.streaming_data else len(train_dataloader)
    save_steps = int(steps_per_epoch * args.save_epoch_percentage)
    total_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=args.num_warmup_steps,
                                                 num_training_steps=total_steps)

    if index == 0:
        print("Starting training...", flush=True)
        print(f"Total steps: {total_steps}", flush=True)
        print(f"Total epochs: {args.num_train_epochs}", flush=True)
        print(f"Total warmup steps: {args.num_warmup_steps}", flush=True)
        print(f"Saving model every: {save_steps}", flush=True)

    # Training loop
    total_step = 0
    for epoch in range(args.num_train_epochs):
        total_loss = 0.
        sub_total_loss = 0.
        sub_step = 0
        model.train()
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            total_loss += loss.item()
            sub_total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # Replaces xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % max(args.logging_steps, args.gradient_accumulation_steps) == 0:
                global_avg_loss = total_loss / step
                global_avg_loss_N = sub_total_loss / sub_step
                perplexity = math.exp(global_avg_loss)
                perplexity_N = math.exp(global_avg_loss_N)
                print(f"Epoch {epoch+1}, step {step}, loss: {global_avg_loss}, train perplexity: {perplexity}")

                sub_step = 0
                sub_total_loss = 0.

                if index == 0:
                    wandb.log({
                        "train_global_average_loss_epoch": global_avg_loss,
                        f"train_global_average_loss_N{args.logging_steps}": global_avg_loss_N,
                        "train_perplexity_epoch": perplexity,
                        f"train_perplexity_epoch_N{args.logging_steps}": perplexity_N,
                        "epoch": epoch,
                        "step": step,
                        "total_step": total_step
                    })

            total_step += 1
            sub_step +=1

            if (index == 0) and (step % save_steps == 0) and (step > 0):
                print("Saving model...")
                if args.output_dir.startswith("gs://"):
                    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                        local_output_dir = tmpdirname
                        print(f"Saving model to {local_output_dir}...")
                        model_cpu = copy.deepcopy(model).to("cpu")
                        model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                        print(f"Uploading model to {args.output_dir}...")
                        subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
                        current_time = datetime.datetime.now().strftime("%Y%m%d")
                        new_dir_name = f"{args.model_name}_epoch{epoch}_step{step}_{current_time}"
                        new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                        subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir), new_gcs_path], check=True)
                else:
                    model_cpu = copy.deepcopy(model).to("cpu")
                    model_cpu.save_pretrained(args.output_dir, save_serialization=True)

        val_ppl,val_loss = evaluate(model, validation_loader, device)
        print(f"Epoch {epoch+1} Validation Perplexity: {val_ppl:.3f}")

        if index == 0:
            wandb.log({
                "val_perplexity": val_ppl,
                "val_eval_loss": val_loss,
                "epoch": epoch,
            })

        # Save model checkpoint at the end of each epoch
        if index == 0:
            print("Saving model...")
            if args.output_dir.startswith("gs://"):
                with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                    local_output_dir = tmpdirname
                    print(f"Saving model to {local_output_dir}...")
                    model_cpu = copy.deepcopy(model).to("cpu")
                    model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                    print(f"Uploading model to {args.output_dir}...")
                    subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
                    current_time = datetime.datetime.now().strftime("%Y%m%d")
                    new_dir_name = f"{args.model_name}_epoch{epoch}_{current_time}"
                    new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                    subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir), new_gcs_path], check=True)
            else:
                model_cpu = copy.deepcopy(model).to("cpu")
                model_cpu.save_pretrained(args.output_dir, save_serialization=True)

    # Finish wandb run
    if index == 0:
        wandb.finish()

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
    parser.add_argument("--num_cores", type=int, default=1)  # Set to 1 for single GPU training
    parser.add_argument("--keep_in_memory", action='store_true')
    parser.add_argument("--streaming_data", action='store_true')
    parser.add_argument("--sharded_data", action='store_true')
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key")
    args = parser.parse_args()


    wandb.login(key=args.wandb_key)

    if args.keep_in_memory and args.streaming_data:
        raise ValueError("keep_in_memory and streaming_data are mutually exclusive. Please set only one of them to True.")

    if args.streaming_data and args.shuffle_buffer_size is None:
        raise ValueError("shuffle_buffer_size is required when streaming_data is set to True.")

    # Set the same seed for reproducibility
    torch.manual_seed(args.seed)
    tokenized_dataset, tokenizer = prep_fn(args)

    args.tokenized_datasets = tokenized_dataset
    args.tokenizer = tokenizer

    # For GPU training, we are using a single process.
    print("Running on GPU (or CPU if no GPU is available)...")
    train_fn(0, args)

if __name__ == "__main__":
    print("Starting...")
    main()
