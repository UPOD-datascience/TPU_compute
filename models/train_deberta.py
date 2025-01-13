print("Importing libraries...")
import os
#os.environ.pop('TPU_PROCESS_ADDRESSES')
#os.environ.pop('CLOUD_TPU_TASK_ID')

import argparse
import torch
import torch_xla
from torch_xla.runtime import global_ordinal
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from tokenizers import ByteLevelBPETokenizer

from transformers import (
    DebertaConfig,
    DebertaForMaskedLM,
    DataCollatorForLanguageModeling,
)

from datasets import load_dataset, load_from_disk

def train_fn(index, args):
    # Each process has its own device (TPU core)
    device = xm.xla_device()
    xm.master_print(f"Process index = {index} started")
    rank = global_ordinal()

    # Load dataset
    xm.master_print("Loading dataset...")
    data_files = {"train": args.train_file}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.data_in_memory==True:
        xm.master_print("Loading dataset..assuming in memory.")
        raw_datasets = load_dataset("json",
                                    data_files=data_files,
                                    num_proc=1,
                                    keep_in_memory=True)
                                    #cache_dir=f'/home/bes3/.cache/huggingface/datasets/json/{rank}')
    else:
        if xm.is_master_ordinal():
            xm.master_print("Loading dataset..assuming from disk.")
            raw_datasets = load_dataset("json",
                                        data_files=data_files,
                                        num_proc=1,
                                        keep_in_memory=False)
            xm.master_print("Saving dataset to disk.")
            raw_datasets.save_to_disk('/home/bes3/.cache/huggingface/datasets/nlllm')

        xm.master_print("Waiting for saving operation to finish.")
        xm.rendezvous("save_to_disk")
        raw_datasets = load_from_disk("/home/bes3/.cache/huggingface/datasets/nlllm")


    # Load tokenizer
    xm.master_print("Loading tokenizer...")
    tokenizer = ByteLevelBPETokenizer.from_file(merges_filename=os.path.join(args.tokenizer_name_or_path, 'merges.txt'),
        vocab_filename=os.path.join(args.tokenizer_name_or_path, 'vocab.json'))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation", None)

    # Create the model on the XLA device
    xm.master_print("Creating the model...")
    config = DebertaConfig()
    model = DebertaForMaskedLM(config).to(device)

    xm.master_print("Creating the data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Dataloaders with Distributed Sampler
    xm.master_print("Creating data loaders...")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator
    )

    # Set up optimizer, etc.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    # Example training loop (simplistic)
    xm.master_print("Staring model training..")
    model.train()
    for epoch in range(args.num_train_epochs):
        xm.master_print(f"Starting epoch {epoch+1}")
        for step, batch in enumerate(train_loader):
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

            optimizer.step()
            optimizer.zero_grad()

            if step % args.logging_steps == 0:
                xm.master_print(f"Epoch {epoch+1}, step {step}, loss: {loss.item()}")

    # Optionally save on master
    if xm.is_master_ordinal():
        xm.master_print("Saving model...")
        if args.output_dir.startswith("gs://"):
            import tempfile
            import subprocess
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_output_dir = tmpdirname
                model.save_pretrained(local_output_dir)
                xm.master_print(f"Uploading model to {args.output_dir}...")
                subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
        else:
            model.save_pretrained(args.output_dir)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--data_in_memory", action='store_true')
    args = parser.parse_args()

    # Set the same seed for all processes
    torch.manual_seed(args.seed)

    # Spawn processes
    xmp.spawn(train_fn, args=(args,), nprocs=8)
    #debug_single_process = args.num_cores == 1
    #torch_xla.launch(
    #    train_fn, args=(args,),
    #    debug_single_process=debug_single_process)


if __name__ == "__main__":
    print("Starting...")
    main()
