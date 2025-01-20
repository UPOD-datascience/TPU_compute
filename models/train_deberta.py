print("Importing libraries...")
import os
import copy
try:
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
except:
    print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

import argparse
import tempfile
import subprocess
import math
import wandb
from numpy import true_divide
import numpy as np
import torch
import torch_xla
from torch_xla.runtime import global_ordinal, world_size
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch.distributed as dist

from tokenizers import ByteLevelBPETokenizer

from transformers import (
    DebertaConfig,
    DebertaForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

def prep_fn(args):
    # Each process has its own device (TPU core)
    #rank = global_ordinal()

    # Load dataset
    print("Loading dataset...")
    raw_datasets = load_from_disk(args.dataset_dir,keep_in_memory=args.keep_in_memory)

    print("Loading tokenizer...")
    _tokenizer = ByteLevelBPETokenizer.from_file(merges_filename=os.path.join(args.tokenizer_name_or_path, 'merges.txt'),
        vocab_filename=os.path.join(args.tokenizer_name_or_path, 'vocab.json'))

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tokenizer._tokenizer, model_max_length=args.max_seq_length, truncation=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '<unk>'})
    tokenizer.add_special_tokens({'mask_token': '<mask>'})

    if args.pre_tokenized == False:
        # Load tokenizer

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length"
            )

        print("Tokenizing datasets...")
        next(iter(raw_datasets["train"]))
        # maybe use forkserver
        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
        )

        print("Getting train and eval datasets...")
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets.get("validation", None)

        return DatasetDict({'train': train_dataset, 'validation': eval_dataset}), tokenizer
    else:
        return raw_datasets, tokenizer


def train_fn(index, args):
    print(f"Process {index} is starting...")

    if xm.is_master_ordinal():
        wandb.login(key='29c3dd3150a673a67772f6b5ea35d0e5d835b0fa')
        wandb.init(
            # set the wandb project where this run will be logged
            project="DeBERTa TPU testing",
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.learning_rate,
            "architecture": "DeBERTa",
            "dataset": "testing",
            "epochs": args.num_train_epochs,
            "weight_decacy": args.weight_decay,
            }
        )


    xm.master_print("Creating the model...")
    device = xm.xla_device()
    # Create the model on the XLA device
    config = DebertaConfig()
    model = DebertaForMaskedLM(config).to(device)

    xm.master_print(f"Model config: {config}")

    xm.master_print("Creating the data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=args.tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Dataloaders with Distributed Sampler
    xm.master_print("Creating training data loader...")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        args.tokenized_datasets["train"],
        num_replicas=world_size(),
        rank=global_ordinal(),
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        args.tokenized_datasets["train"],
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator
    )
    xla_train_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())

    if args.tokenized_datasets["validation"] is not None:
        xm.master_print("Creating validation data loader...")
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            args.tokenized_datasets["validation"],
            num_replicas=world_size(),
            rank=global_ordinal(),
            shuffle=False
        )
        validation_loader = torch.utils.data.DataLoader(
            args.tokenized_datasets["validation"],
            batch_size=args.per_device_train_batch_size,
            sampler=validation_sampler,
            collate_fn=data_collator
        )
        xla_val_loader = pl.MpDeviceLoader(validation_loader, xm.xla_device())

    # Set up optimizer, etc.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # maybe use standard Training functions from transformers library..
    # Example training loop (simplistic)
    xm.master_print("Starting model training..")
    model.train()
    total_step = 0
    for epoch in range(args.num_train_epochs):
        total_loss = 0.
        xm.master_print(f"Starting epoch {epoch+1}")
        for step, batch in enumerate(xla_train_loader):
            # Move batch to the XLA device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)
            #optimizer.step()
            optimizer.zero_grad()

            total_loss+=loss.item()
            total_step += 1
            if step % args.logging_steps == 0:
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

        if args.tokenized_datasets["validation"] is not None:
            val_ppl = evaluate(model, xla_val_loader, device)
            xm.master_print(f"Epoch {epoch+1} Validation Perplexity: {val_ppl:.3f}")
            if xm.is_master_ordinal():
                wandb.log({
                    "validation_perplexity": val_ppl,
                    "epoch": epoch
                })

    # Optionally save on master
    if xm.is_master_ordinal():
        xm.master_print("Saving model...")
        if args.output_dir.startswith("gs://"):
            with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                local_output_dir = tmpdirname
                xm.master_print(f"Saving model to {local_output_dir}...")
                model_cpu = copy.deepcopy(model).to("cpu")
                model_cpu.save_pretrained(local_output_dir, save_serialization=True)
                xm.master_print(f"Uploading model to {args.output_dir}...")
                subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
        else:
            model_cpu = copy.deepcopy(model).to("cpu")
            model_cpu.save_pretrained(args.output_dir, save_serialization=True)

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
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--keep_in_memory", action='store_true')
    parser.add_argument("--streaming_data", action='store_true')
    args = parser.parse_args()

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
    # debug_single_process = args.num_cores == 1
    # print(f"debug_single_process: {debug_single_process}")
    # torch_xla.launch(
    #     train_fn, args=(args,),
    #     debug_single_process=debug_single_process)


if __name__ == "__main__":
    print("Starting...")
    main()
