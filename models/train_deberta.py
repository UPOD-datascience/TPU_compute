"""
This is the main script to continue pre-training a DeBERTa model.
"""

import argparse
import os
import sys
import traceback

import torch

os.environ["HF_DATASETS_CACHE"] = "/home/bes3/datasets_cache"
os.environ["HF_HOME"] = "/home/bes3/hf_cache"

# Ensure cache directories exist
os.makedirs("/home/bes3/datasets_cache", exist_ok=True)
os.makedirs("/home/bes3/hf_cache", exist_ok=True)

import shutil

cache_dirs = [
    os.path.expanduser("~/.cache/huggingface"),
    os.path.expanduser("~/.cache/datasets"),
    "/tmp/datasets",
]
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        print(f"Removing cache directory: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir)

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.runtime import global_ordinal, world_size

    print("XLA import successful")
except ImportError as e:
    print(f"XLA import failed: {e}")
    exit(1)

import copy
import datetime
import gc
import io
import math
import random
import subprocess
import tempfile
from functools import partial
from itertools import chain
from time import sleep

import fsspec
import numpy as np
import wandb
from datasets import DatasetDict, DatasetInfo, load_dataset
from gcsfs import GCSFileSystem
from google.cloud import storage
from lazy_grouping import LazyGroupingDataset
from safetensors.torch import load_file as load_safetensors
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2TokenizerFast,
    Trainer,
    TrainingArguments,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    os.environ.pop("TPU_PROCESS_ADDRESSES")
    os.environ.pop("CLOUD_TPU_TASK_ID")
except:
    print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

print("TPU_NUM_DEVICES:", os.environ.get("TPU_NUM_DEVICES"), flush=True)
print(
    "TPU_CHIPS_PER_HOST_BOUNDS:",
    os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS"),
    flush=True,
)

worker_id = int(os.environ.get("TPU_WORKER_ID", "0"))
# Determine total number of shards (for example, from the comma-separated hostnames)
shards = os.environ.get("TPU_WORKER_HOSTNAMES", "0").split(",")

print(f"TPU_WORKER_ID: {worker_id}")
print(f"TPU_WORKER_HOSTNAMES for {worker_id}: {shards}")


def _barrier(tag, device):
    """Cross-host barrier using mesh_reduce (true ICI all-reduce).

    xm.rendezvous() may only synchronise processes on the *same* host on
    multi-host TPU pods.  mesh_reduce performs a real all-reduce over the
    ICI interconnect, so every rank across every host must participate
    before any of them can proceed.
    """
    dummy = torch.tensor([0.0], device=device)
    xm.mesh_reduce(tag, dummy, lambda x: sum(x))


def shuffle_and_save_dataset(dataset, output_path, seed=None, shuffle=True):
    # Shuffle only the training dataset
    if shuffle:
        shuffled_train = dataset["train"].shuffle(seed=seed)
    else:
        shuffled_train = dataset["train"]

    # Save the shuffled training dataset and the original validation dataset
    shuffled_dataset = DatasetDict(
        {"train": shuffled_train, "validation": dataset["validation"]}
    )
    print("Saving shuffled data to disk", flush=True)
    shuffled_dataset.save_to_disk(output_path)
    print(f"Shuffled dataset saved to {output_path}", flush=True)


class ShardedShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        num_shards,
        shard_id,
        shuffle_buffer_size,
        max_steps=None,
        batch_size=1,
    ):
        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_steps = max_steps
        self.batch_size = batch_size

    def __iter__(self):
        buffer = []
        items_yielded = 0
        max_items = (
            self.max_steps * self.batch_size
            if self.max_steps is not None
            else float("inf")
        )

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
    # Tokenize without truncation first; chunking is handled by group_texts
    return tokenizer(
        examples["text"],
        truncation=False,
        max_length=max_seq_length,
        padding="max_length",
    )


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
    if local_path.endswith(".safetensors"):
        checkpoint = load_safetensors(local_path, device="cpu")
        checkpoint = {k: v.to(xm.xla_device()) for k, v in checkpoint.items()}
    else:
        checkpoint = torch.load(local_path, map_location=device)

    return checkpoint


def load_sharded_dataset(datasets, dformat, args):
    # For single process mode, use process-based indices instead of TPU core indices
    if args.num_cores == 1:
        num_shards = 1
        shard_idx = 0
    else:
        num_shards = max(world_size(), 16)
        shard_idx = global_ordinal()

    print(f"Sharding for shard index:{shard_idx} / {num_shards}")

    dataset_dict = load_dataset(
        dformat, data_files=datasets, streaming=False, keep_in_memory=True
    )

    # Apply sharding to each split
    sharded_dataset = {}
    for split in dataset_dict.keys():
        sharded_dataset[split] = dataset_dict[split].shard(
            num_shards=num_shards, index=shard_idx
        )

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
        num_chunks = (
            example_length + max_seq_length - 1
        ) // max_seq_length  # Ceiling division

        # Split each feature into chunks
        for k, tokens in current_example.items():
            # Create chunks of max_seq_length
            chunks = []
            for j in range(0, example_length, max_seq_length):
                chunk = tokens[j : min(j + max_seq_length, example_length)]

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
    # For streaming data, each process needs to load and tokenize independently
    # but we stagger the start times to avoid resource contention
    current_ordinal = global_ordinal()
    stagger_delay = current_ordinal * 2  # 2 seconds per core

    if stagger_delay > 0:
        print(
            f"Process {current_ordinal}: Waiting {stagger_delay}s to stagger dataset loading..."
        )
        sleep(stagger_delay)

    # Load and tokenize dataset
    if args.pre_tokenized:
        if args.training_file and args.validation_file:
            datasets = {"train": args.training_file, "validation": args.validation_file}
        else:
            datasets = {
                "train": args.dataset_dir
                + f"/train_{args.max_seq_length}.{args.dataset_format}",
                "validation": args.dataset_dir
                + f"/validation_{args.max_seq_length}.{args.dataset_format}",
            }
        if args.streaming_data:
            print(
                f"Process {current_ordinal}: Loading pre-tokenized streaming dataset..",
                flush=True,
            )
            tokenized_dataset = load_dataset(
                args.dataset_format,
                data_files=datasets,
                streaming=True,
                keep_in_memory=False,
            ).shuffle(buffer_size=args.shuffle_buffer_size)
        else:
            print(
                f"Process {current_ordinal}: Loading pre-tokenized dataset, no streaming",
                flush=True,
            )
            tokenized_dataset = load_dataset(
                args.dataset_format,
                data_files=datasets,
                streaming=False,
                keep_in_memory=True,
            )
    else:
        print(
            f"Process {current_ordinal}: Tokenizing dataset... with streaming: {args.streaming_data} and keep_in_memory: {args.keep_in_memory}",
            flush=True,
        )
        print(f"Dataset location: {args.dataset_dir}", flush=True)

        train_loc = "train" if not args.debug else "validation"

        if args.training_file and args.validation_file:
            datasets = {"train": args.training_file, "validation": args.validation_file}
        else:
            datasets = {
                "train": args.dataset_dir + f"/{train_loc}/*.{args.dataset_format}",
                "validation": args.dataset_dir + f"/validation/*.{args.dataset_format}",
            }

        if args.streaming_data:
            # Estimate steps/epoch only when loading from a directory
            if (not args.training_file) and (not args.validation_file):
                try:
                    if current_ordinal == 0:  # Only master process calculates this
                        ds_train_info = DatasetInfo.from_directory(
                            args.dataset_dir + f"/{train_loc}/"
                        )
                        num_examples = ds_train_info["num_examples"]
                        args.max_steps_per_epoch = (
                            num_examples
                            // args.per_device_train_batch_size
                            // max(world_size(), 1)
                        )
                        print(
                            f"Maximum steps per epoch: {args.max_steps_per_epoch}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"Could not obtain datasetinfo:{e}")
                    pass

            print(f"Process {current_ordinal}: Init streaming dataset...", flush=True)
            dataset = load_dataset(
                args.dataset_format,
                data_files=datasets,
                streaming=True,
                keep_in_memory=False,
            )
        else:
            print(
                f"Process {current_ordinal}: Init non-streaming dataset...", flush=True
            )
            if args.sharded_data:
                print("Sharding data...", flush=True)
                dataset = load_sharded_dataset(datasets, args.dataset_format, args)
            else:
                dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=False,
                    keep_in_memory=True,
                )

        print(f"Process {current_ordinal}: Tokenizing dataset...", flush=True)
        tokenize_fn = partial(
            tokenize_function,
            tokenizer=args.tokenizer,
            max_seq_length=args.max_seq_length,
        )

        # Compute safe remove_columns based on actual columns present in dataset/split
        candidate_remove = [
            "text",
            "id",
            "source",
            "approx_token_counts_translated",
            "approx_token_counts_original",
        ]
        try:
            colnames = dataset["train"].column_names
        except Exception:
            try:
                colnames = dataset.column_names
            except Exception:
                colnames = []
        safe_remove = [c for c in candidate_remove if c in (colnames or [])]

        if args.streaming_data:
            tokenized_dataset_raw = dataset.map(
                tokenize_fn, batched=True, remove_columns=safe_remove
            )
        else:
            tokenized_dataset_raw = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=safe_remove,
                num_proc=1,  # Use single process to avoid conflicts
            )

        # If lazy grouping is enabled, return the raw tokenized dataset
        if args.lazy_grouping:
            print(
                "Lazy grouping enabled. Grouping will be performed on-the-fly during training.",
                flush=True,
            )
            print(
                f"This will reduce startup time but may slightly impact training speed.",
                flush=True,
            )
            return tokenized_dataset_raw

        # Otherwise, perform grouping now as before
        print(
            f"Process {current_ordinal}: Performing chunking tokenized data...",
            flush=True,
        )
        group_fn = partial(
            group_texts,
            max_seq_length=args.max_seq_length,
            pad_token=args.tokenizer.pad_token_id,
        )
        if args.streaming_data:
            tokenized_dataset = tokenized_dataset_raw.map(group_fn, batched=True)
        else:
            tokenized_dataset = tokenized_dataset_raw.map(
                group_fn,
                batched=True,
                num_proc=1,
                desc=f"Grouping texts in chunks of {args.max_seq_length}",
            )
        del tokenized_dataset_raw

    print(f"Process {current_ordinal}: Dataset preparation complete", flush=True)

    # Optional: Add a barrier to ensure all processes finish around the same time
    _barrier("dataset_prep_complete", xm.xla_device())

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
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            # Don't continue on XLA tensor errors - these need to be fixed
            if "XLA" in str(e) or "tensor" in str(e).lower():
                print("XLA/tensor error detected - stopping iteration")
                raise
            continue


def train_fn(tokenized_dataset, device, args):
    # Initialize wandb for the master process
    print(f"Process {global_ordinal()}: Starting train_fn...", flush=True)
    if global_ordinal() == 0:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="DeBERTaV2 TPU pretraining from scratch",
            config={
                "TPU ID": args.TPU_NAME,
                "TPU DISK": args.TPU_DISK,
                "learning_rate": args.learning_rate,
                "architecture": "DeBERTaV2",
                "dataset": args.dataset_dir,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
                "lazy_grouping": args.lazy_grouping,
            },
            mode="online",
            dir="/home/bes3/temp",
        )
        wandb.run.log_code(root="/home/bes3/models", name="train_deberta")

    # Load model configuration
    # config = RobertaConfig.from_pretrained(args.model_name)

    # Load pre-trained model
    print(f"Process {global_ordinal()}: Loading the LM...", flush=True)

    if args.checkpoint_handling in ["start_with_checkpoint", "start_with_init"]:
        model_config = DebertaV2Config.from_pretrained(args.model_name)
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

        model = DebertaV2ForMaskedLM(model_config)

        if isinstance(args.checkpoint_path, str) & (args.checkpoint_path != ""):
            if args.checkpoint_path.startswith("gs://"):
                # Parse GCS path
                bucket_name = args.checkpoint_path.split("/")[2]
                blob_name = "/".join(args.checkpoint_path.split("/")[3:])
                local_path = f"/tmp/checkpoint.{args.checkpoint_path.split('.')[-1]}"  # Temporary local path to store the downloaded file
                checkpoint = load_from_gcs(bucket_name, blob_name, local_path, device)
                sleep(1)
                print(
                    f"Process {global_ordinal()}: Checkpoint downloaded...", flush=True
                )
                _barrier("checkpoint_downloaded", device)
            else:
                if args.checkpoint_path.endswith(".safetensors"):
                    checkpoint = load_safetensors(args.checkpoint_path, device=device)
                else:
                    checkpoint = torch.load(args.checkpoint_path, map_location=device)
                sleep(1)
                print(
                    f"Process {global_ordinal()}: Checkpoint downloaded...", flush=True
                )
                _barrier("checkpoint_downloaded", device)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            sleep(1)
            print(
                f"Process {global_ordinal()}: Checkpoint loaded in memory..", flush=True
            )
        else:
            pass
    elif args.checkpoint_handling == "start_with_basemodel":
        model = DebertaV2ForMaskedLM.from_pretrained(args.model_name)
    else:
        raise ValueError(
            "Checkpoint handling should be one of ['start_with_init','start_with_base', 'start_with_checkpoint']"
        )

    _barrier("checkpoint_loaded_in_memory", device)

    # Move model to device — stagger across ranks to avoid OOM spikes
    print(
        f"Process {global_ordinal()}: world_size={world_size()}, global_ordinal={global_ordinal()}, num_cores={args.num_cores}",
        flush=True,
    )

    if args.num_cores == 1:
        model = model.to(device=device, dtype=torch.bfloat16)
        xm.mark_step()
    else:
        for i in range(world_size()):
            if global_ordinal() == i:
                print(
                    f"Process {global_ordinal()}: My turn to load model to TPU device {device}...",
                    flush=True,
                )
                model = model.to(device=device, dtype=torch.bfloat16)
                print(
                    f"Process {global_ordinal()}: Model loaded on device successfully",
                    flush=True,
                )
            _barrier(f"model_to_device_{i}", device)

    print(f"Process {global_ordinal()}: All models loaded to devices", flush=True)

    # Set up data collator
    print(f"Process: {global_ordinal()}. Setting up data collator.")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=args.tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    _barrier("datacollator", device)

    # Decide on distributed sampler parameters:
    if args.num_cores == 1:
        sampler_rank = 0
        sampler_replicas = 1
        print(f"Process {global_ordinal()}: Using single core mode", flush=True)
    else:
        sampler_rank = global_ordinal()
        sampler_replicas = world_size()

    print(f"Process {global_ordinal()}: About to create DataLoader...", flush=True)

    # Create sampler
    distributed_sampler = False
    if args.streaming_data:
        print(f"Process {global_ordinal()}: Creating sharded dataset...", flush=True)

        if args.num_cores == 1:
            num_shards = 1
            shard_id = 0
        else:
            num_shards = world_size()
            shard_id = global_ordinal()
        print(
            f"Process {global_ordinal()}: Num shards: {num_shards}, shard id: {shard_id}",
            flush=True,
        )

        try:
            sharded_train_dataset = tokenized_dataset["train"].shard(
                num_shards=num_shards, index=shard_id
            )
            print(
                f"Process {global_ordinal()}: Sharded dataset created successfully",
                flush=True,
            )
        except Exception as e:
            print(
                f"Process {global_ordinal()}: Error creating sharded dataset: {e}",
                flush=True,
            )
            print(
                f"Process {global_ordinal()}: Falling back to index-based streaming filter sharding...",
                flush=True,
            )
            sharded_train_dataset = tokenized_dataset["train"].filter(
                lambda ex, idx: idx % num_shards == shard_id, with_indices=True
            )

        # Apply lazy grouping if enabled
        if args.lazy_grouping and not args.pre_tokenized:
            print(
                f"Applying lazy grouping to sharded dataset for worker {shard_id}...",
                flush=True,
            )
            _sharded_train_dataset = LazyGroupingDataset(
                sharded_train_dataset,
                max_seq_length=args.max_seq_length,
                pad_token=args.tokenizer.pad_token_id,
                batch_size=args.per_device_train_batch_size,
                streaming=args.streaming_data,
            )
        else:
            _sharded_train_dataset = sharded_train_dataset

        print(
            f"Process {global_ordinal()}: Creating streaming DataLoader...", flush=True
        )

        train_dataloader = torch.utils.data.DataLoader(
            _sharded_train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
    else:
        xm.master_print("Starting the DistributedSampler...")
        distributed_sampler = True
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            tokenized_dataset["train"],
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            shuffle=True,
        )
        # Create dataloaders
        xm.master_print("Starting the DataLoader...")
        train_dataloader = torch.utils.data.DataLoader(
            tokenized_dataset["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            sampler=train_sampler,
        )
    #################
    validation_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )

    if not args.streaming_data:
        del tokenized_dataset
        gc.collect()

    if args.num_cores == 1:
        xla_train_loader = train_dataloader
        xla_validation_loader = validation_dataloader
    else:
        xla_train_loader = pl.MpDeviceLoader(train_dataloader, xm.xla_device())
        xla_validation_loader = pl.MpDeviceLoader(
            validation_dataloader, xm.xla_device()
        )

    print(f"XLA device is: {xm.xla_device()}", flush=True)
    print(f"Device for training: {device}", flush=True)

    _barrier("data_loader", device)

    # Set up optimizer and scheduler AFTER model is on device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = (
        args.max_steps_per_epoch if args.streaming_data else len(train_dataloader)
    )
    save_steps = (
        int(steps_per_epoch * args.save_epoch_percentage)
        if args.save_epoch_percentage < 1
        else int(args.save_epoch_percentage)
    )
    # Align save_steps to the nearest accumulation boundary
    _gas = args.gradient_accumulation_steps
    if _gas > 1 and save_steps % _gas != 0:
        save_steps = max(_gas, (save_steps // _gas) * _gas)
    save_steps = max(save_steps, _gas)

    # Align logging_steps to accumulation boundaries
    if _gas > 1 and args.logging_steps % _gas != 0:
        aligned_logging = max(_gas, (args.logging_steps // _gas) * _gas)
        print(
            f"[WARNING] logging_steps={args.logging_steps} is not a multiple of "
            f"gradient_accumulation_steps={_gas}. Aligning to {aligned_logging}.",
            flush=True,
        )
        args.logging_steps = aligned_logging

    total_steps = steps_per_epoch * args.num_train_epochs

    if global_ordinal() == 0:
        print(
            f"[CONFIG] steps_per_epoch={steps_per_epoch}, save_steps={save_steps} "
            f"(aligned to GAC={_gas}), logging_steps={args.logging_steps}, "
            f"total_steps={total_steps}",
            flush=True,
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_steps,
    )
    # Use cyclic scheduler with hard restarts (cosine with restarts)
    # Calculate number of cycles - using 1 cycle per epoch as default, or user-specified value
    # num_cycles = args.num_cycles if args.num_cycles is not None else args.num_train_epochs
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=total_steps,
    #     num_cycles=num_cycles
    # )

    _barrier("start_training", device)
    if global_ordinal() == 0:
        print("Starting training...", flush=True)
        print(f"Total steps: {total_steps}", flush=True)
        print(f"Total epochs: {args.num_train_epochs}", flush=True)
        print(f"Total warmup steps: {args.num_warmup_steps}", flush=True)

    # Training loop
    total_step = 0
    miss_steps = 0
    print(
        f"ENTERING THE TRAINING LOOP with: {args.num_train_epochs} epochs", flush=True
    )

    for epoch in range(args.num_train_epochs):
        _barrier(f"start_epoch_{epoch}", device)
        print(
            f"Starting with epoch {epoch}...for process {global_ordinal()}", flush=True
        )

        total_loss = (
            0.0  # kept as XLA tensor during accumulation, materialised at logging
        )
        sub_total_loss = 0.0
        sub_step = 0
        model.train()
        if distributed_sampler:
            print(f"Starting with epoch {epoch}...", flush=True)
            train_sampler.set_epoch(epoch)
            print("done with setting epoch..", flush=True)

        print(f"Entering data loader iterator for epoch {epoch}...", flush=True)
        step = -1

        # Use a while-loop instead of for-loop so that when one core's data
        # shard is exhausted it keeps participating in collective operations
        # (optimizer_step, mesh_reduce, barriers) until ALL cores agree to
        # stop.  This prevents the cross-deadlock that occurs when one core
        # exits the loop while others block on a collective op.
        data_iter = iter(safe_iter(xla_train_loader))
        local_exhausted = False
        DIAG_START = 5990  # start verbose logging this many steps before expected hang

        while True:
            step += 1
            _diag = step >= DIAG_START

            # --- fetch next batch (or mark exhausted) ---
            batch = None
            if not local_exhausted:
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} fetching batch…",
                        flush=True,
                    )
                try:
                    batch = next(data_iter)
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} batch OK",
                            flush=True,
                        )
                except StopIteration:
                    local_exhausted = True
                    print(
                        f"Process {global_ordinal()}: Data shard exhausted at step {step}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"Process {global_ordinal()}: Error fetching batch at step {step}: {e}",
                        flush=True,
                    )
                    if "XLA" in str(e) or "tensor" in str(e).lower():
                        raise
                    local_exhausted = True

            # --- at every gradient-accumulation boundary, check whether
            #     ANY core is done. ---
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} ENTER epoch_stop_check (exhausted={local_exhausted})",
                        flush=True,
                    )
                flag = torch.tensor([1.0 if local_exhausted else 0.0], device=device)
                any_done = xm.mesh_reduce("epoch_stop_check", flag, max)
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} EXIT epoch_stop_check any_done={any_done.item()}",
                        flush=True,
                    )
                if any_done.item() > 0.5:
                    print(
                        f"Process {global_ordinal()}: All cores stopping "
                        f"at step {step} (at least one shard exhausted)",
                        flush=True,
                    )
                    break  # all cores break here together

            # --- exhausted core: still participate in every collective op ---
            if local_exhausted:
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} exhausted-path",
                        flush=True,
                    )
                optimizer.zero_grad()
                step_ok = False
            else:
                # --- normal training step ---
                assert batch is not None, "batch should never be None here"

                step_ok = True
                try:
                    if step < 3:
                        print(
                            f"Process {global_ordinal()}: Processing batch {step}...",
                            flush=True,
                        )

                    batch = {
                        k: v.to(device=device, dtype=torch.bfloat16)
                        if v.dtype == torch.float32
                        else v.to(device)
                        for k, v in batch.items()
                    }
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps

                    # Unconditionally replace NaN/Inf — no data-dependent branch.
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                    loss.backward()

                    # Accumulate loss as XLA tensor (avoid .item() every micro-step)
                    total_loss += loss.detach() * args.gradient_accumulation_steps
                    sub_total_loss += loss.detach() * args.gradient_accumulation_steps

                except Exception as e:
                    print(
                        f"Error occurred during processing batch {step}: {e}",
                        flush=True,
                    )
                    print("Full traceback:")
                    traceback.print_exc()
                    miss_steps += 1
                    step_ok = False
                    optimizer.zero_grad()

            # ══════════════════════════════════════════════════════
            # ── Collective operations — ALWAYS executed ───────────
            # Both the exhausted-path and the normal-training path
            # converge here. These are outside try/except so even
            # if one rank had an exception, it still participates.
            # ══════════════════════════════════════════════════════

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} ENTER grad_clip",
                        flush=True,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} ENTER optimizer_step",
                        flush=True,
                    )
                xm.optimizer_step(optimizer, barrier=True)
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} EXIT optimizer_step",
                        flush=True,
                    )
                scheduler.step()
                xm.mark_step()
                optimizer.zero_grad()

                total_step += args.gradient_accumulation_steps

                # ══════════════════════════════════════════════════
                # ── Logging & saving — ONLY at accumulation boundaries ──
                # All collective ops (mesh_reduce, barriers) MUST be
                # placed here, right after optimizer_step + mark_step,
                # where all ranks are provably synchronized.
                # ══════════════════════════════════════════════════

                # --- Logging ---
                if (step + 1) % args.logging_steps == 0:
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} ENTER mesh_reduce(loss)",
                            flush=True,
                        )
                    total_loss_val = (
                        total_loss.item()
                        if isinstance(total_loss, torch.Tensor)
                        else float(total_loss)
                    )
                    sub_total_loss_val = (
                        sub_total_loss.item()
                        if isinstance(sub_total_loss, torch.Tensor)
                        else float(sub_total_loss)
                    )
                    total_loss = total_loss_val
                    sub_total_loss = sub_total_loss_val

                    local_avg_loss = total_loss_val / max(step + 1, 1)
                    local_avg_loss_N = sub_total_loss_val / max(sub_step, 1)

                    global_avg_loss = xm.mesh_reduce(
                        f"loss_s{step}", local_avg_loss, np.mean
                    )
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} EXIT mesh_reduce(loss)",
                            flush=True,
                        )
                    global_avg_loss_N = xm.mesh_reduce(
                        f"loss_N_s{step}", local_avg_loss_N, np.mean
                    )
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} EXIT mesh_reduce(loss_N)",
                            flush=True,
                        )

                    perplexity = math.exp(min(global_avg_loss, 10))
                    perplexity_N = math.exp(min(global_avg_loss_N, 10))

                    sub_step = 0
                    sub_total_loss = 0.0

                    _barrier(f"before_wandb_log_s{step}", device)
                    if global_ordinal() == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(
                            f"Logging for epoch: {epoch}, step: {step}, train_perplexity_N:{perplexity_N}, lr: {current_lr:.2e}",
                            flush=True,
                        )
                        wandb.log(
                            {
                                "train_global_average_loss": global_avg_loss,
                                "train_global_average_loss_N": global_avg_loss_N,
                                "train_perplexity": perplexity,
                                "train_perplexity_N": perplexity_N,
                                "learning_rate": current_lr,
                                "epoch": epoch,
                                "step": step,
                                "total_step": total_step,
                            },
                            commit=True,
                        )
                    _barrier(f"after_wandb_log_s{step}", device)

                # --- Mid-epoch checkpoint saving ---
                # Uses total_step so condition fires only at accumulation
                # boundaries where ranks are synced.
                if save_steps > 0 and total_step % save_steps == 0:
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} ENTER barrier(before_model_saving)",
                            flush=True,
                        )
                    _barrier(f"before_model_saving_s{step}", device)
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} EXIT barrier(before_model_saving)",
                            flush=True,
                        )
                    if global_ordinal() == 0:
                        print(f"Saving model at total_step={total_step}...", flush=True)

                        wandb.log({"epoch": epoch, "step": total_step}, commit=True)

                        if args.output_dir.startswith("gs://"):
                            with tempfile.TemporaryDirectory(
                                dir=args.tmp_dir
                            ) as tmpdirname:
                                local_output_dir = tmpdirname
                                print(
                                    f"Saving model to {local_output_dir}...", flush=True
                                )
                                model_cpu = copy.deepcopy(model).to("cpu")
                                model_cpu.save_pretrained(
                                    local_output_dir, safe_serialization=True
                                )
                                print(
                                    f"Uploading model to {args.output_dir}...",
                                    flush=True,
                                )
                                subprocess.run(
                                    [
                                        "gsutil",
                                        "-m",
                                        "cp",
                                        "-r",
                                        local_output_dir,
                                        args.output_dir,
                                    ],
                                    check=True,
                                )
                                new_dir_name = f"{args.model_name}_latest"
                                new_gcs_path = f"{args.output_dir}/{new_dir_name}"
                                uploaded_gcs_dir = f"{args.output_dir}/{os.path.basename(local_output_dir)}"
                                print(
                                    f"Renaming {uploaded_gcs_dir} → {new_gcs_path}",
                                    flush=True,
                                )
                                subprocess.run(
                                    [
                                        "gsutil",
                                        "-m",
                                        "mv",
                                        uploaded_gcs_dir,
                                        new_gcs_path,
                                    ],
                                    check=True,
                                )
                        else:
                            model_cpu = copy.deepcopy(model).to("cpu")
                            model_cpu.save_pretrained(
                                args.output_dir, safe_serialization=True
                            )
                    _barrier(f"after_model_saving_s{step}", device)
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} EXIT barrier(after_model_saving)",
                            flush=True,
                        )

            if args.debug:
                break

            if step_ok:
                sub_step += 1

            if (step + 1) % steps_per_epoch == 0:
                break

        # All cores arrive here together
        _barrier(f"end_epoch_{epoch}_before_eval", device)

        val_ppl = evaluate(model, xla_validation_loader, device)

        if global_ordinal() == 0:
            wandb.log(
                {"val_perplexity": val_ppl, "epoch": epoch, "miss_steps": miss_steps},
                commit=True,
            )

        # Save model checkpoint
        if global_ordinal() == 0:
            print("Saving model...", flush=True)
            if args.output_dir.startswith("gs://"):
                with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                    local_output_dir = tmpdirname
                    print(f"Saving model to {local_output_dir}...", flush=True)
                    model_cpu = copy.deepcopy(model).to("cpu")
                    model_cpu.save_pretrained(local_output_dir, safe_serialization=True)
                    print(f"Uploading model to {args.output_dir}...", flush=True)
                    subprocess.run(
                        ["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir],
                        check=True,
                    )
                    current_time = datetime.datetime.now().strftime("%Y%m%d")
                    new_dir_name = f"{args.model_name}_epoch{epoch}_{current_time}"
                    new_gcs_path = f"{args.output_dir}/{new_dir_name}"
                    uploaded_gcs_dir = (
                        f"{args.output_dir}/{os.path.basename(local_output_dir)}"
                    )
                    print(f"Renaming {uploaded_gcs_dir} → {new_gcs_path}", flush=True)
                    subprocess.run(
                        ["gsutil", "-m", "mv", uploaded_gcs_dir, new_gcs_path],
                        check=True,
                    )
            else:
                model_cpu = copy.deepcopy(model).to("cpu")
                model_cpu.save_pretrained(args.output_dir, safe_serialization=True)

        # Epoch-end save barrier — all ranks must wait for rank 0's save
        _barrier(f"end_epoch_{epoch}_after_save", device)

    # Finish wandb run
    if global_ordinal() == 0:
        wandb.finish()


def prep_and_train_fn(index, args):
    device = xm.xla_device()

    # Verify all cores are participating
    world_size_val = world_size()
    ordinal = global_ordinal()

    print(
        f"Process {index}: ordinal={ordinal}, world_size={world_size_val}, device={device}",
        flush=True,
    )

    _barrier("prep_start", device)
    tokenized_dataset = prep_fn(args)

    _barrier("train_start", device)
    train_fn(tokenized_dataset, device, args)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    xm.mark_step()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        loss = outputs.loss
        total_loss += loss.item()
        total_steps += 1

    local_avg_loss = total_loss / max(total_steps, 1)

    global_avg_loss = xm.mesh_reduce("eval_loss", local_avg_loss, np.mean)

    ppl = math.exp(min(global_avg_loss, 10))

    model.train()
    return ppl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--training_file",
        type=str,
        default=None,
        help="Path to training parquet/json file (local path or gs://)",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Path to validation parquet/json file (local path or gs://)",
    )
    parser.add_argument(
        "--dataset_format", default="json", choices=["json", "parquet", "arrow"]
    )
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pre_tokenized", action="store_true", default=False)
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
    parser.add_argument("--max_grad_norm", type=float, default=14.4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_epoch_percentage", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--keep_in_memory", action="store_true")
    parser.add_argument("--streaming_data", action="store_true")
    parser.add_argument("--sharded_data", action="store_true")
    parser.add_argument("--max_steps_per_epoch", type=int, default=50_000_000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument(
        "--shuffle_dataset_path", type=str, default="/home/bob/tmp/shuffle.parquet"
    )
    parser.add_argument("--shuffle_dataset_ext", type=str, default=None)
    parser.add_argument("--shuffle_dataset", action="store_true")
    parser.add_argument("--shuffle_force_update", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument(
        "--checkpoint_handling",
        type=str,
        choices=["start_with_checkpoint", "start_with_basemodel", "start_with_init"],
    )
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument(
        "--lazy_grouping",
        action="store_true",
        help="Use lazy grouping to process data on-the-fly during training (incompatible with --pre_tokenized)",
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=4,
        help="Number of cycles for cyclic learning rate scheduler. If None, defaults to num_train_epochs",
    )
    parser.add_argument(
        "--wandb_key", type=str, required=True, help="Weights & Biases API key"
    )
    parser.add_argument(
        "--TPU_NAME", type=str, default="unknown, please set with --TPU_NAME"
    )
    parser.add_argument(
        "--TPU_DISK", type=str, default="unknown, please set with --TPU_DISK"
    )
    args = parser.parse_args()

    # give error if both keep_in_memory and streaming_data are set to True
    # as they are mutually exclusive
    if args.keep_in_memory and args.streaming_data:
        raise ValueError(
            "keep_in_memory and streaming_data are mutually exclusive. Please set only one of them to True."
        )

    if args.streaming_data == True and args.shuffle_buffer_size is None:
        raise ValueError(
            "shuffle_buffer_size is required when streaming_data is set to True."
        )

    if args.lazy_grouping and args.pre_tokenized:
        raise ValueError(
            "lazy_grouping cannot be used with pre_tokenized data. Lazy grouping requires raw tokenized data to group on-the-fly."
        )

    # Set the same seed for all processes
    torch.manual_seed(args.seed)

    if args.shuffle_dataset:
        shuffle_dir = args.shuffle_dataset_path
        if (os.path.exists(shuffle_dir)) and (args.shuffle_force_update == False):
            print(
                f"Shuffled dataset exists, skipping creation: {shuffle_dir}", flush=True
            )
        else:
            if os.path.exists(shuffle_dir):
                print(f"Removing shuffled dataset: {shuffle_dir}", flush=True)
                shutil.rmtree(shuffle_dir)

            if args.shuffle_dataset_ext is not None:
                print("Loading pre-shuffled dataset...", flush=True)
                dataset = load_dataset(
                    args.dataset_format,
                    data_files={
                        "train": args.shuffle_dataset_ext,
                        "validation": args.dataset_dir
                        + f"/validation/*.{args.dataset_format}",
                    },
                    keep_in_memory=True,
                )

                dataset.save_to_disk(args.shuffle_dataset_path)

            else:
                print("Loading dataset for shuffling...", flush=True)
                dataset = load_dataset(
                    args.dataset_format,
                    data_files={
                        "train": args.dataset_dir + f"/train/*.{args.dataset_format}",
                        "validation": args.dataset_dir
                        + f"/validation/*.{args.dataset_format}",
                    },
                    keep_in_memory=True,
                )

                print("Shuffling and saving dataset...", flush=True)
                shuffle_and_save_dataset(dataset, args.shuffle_dataset_path)

            # Clear the dataset from memory
            del dataset
            gc.collect()
            print("Cleared shuffled dataset from master's memory", flush=True)
    else:
        shuffle_dir = args.shuffle_dataset_path
        if os.path.exists(shuffle_dir):
            print(f"Removing shuffled dataset: {shuffle_dir}", flush=True)
            shutil.rmtree(shuffle_dir)

    args.tokenizer = DebertaV2TokenizerFast.from_pretrained(args.tokenizer_name_or_path)
    args.tokenizer.model_max_length = args.max_seq_length

    if args.shuffle_dataset:
        args.dataset_dir = args.shuffle_dataset_path
        args.dataset_format = "arrow"
        print("Continuing with Arrow format", flush=True)

    sleep(5)

    # Spawn processes
    print(f"Initializing TPU...with {args.num_cores} cores")
    print("Spawning processes...")

    if args.num_cores == 1:
        print("Running single process...", flush=True)
        prep_and_train_fn(0, args)
    else:
        os.environ["TPU_NUM_DEVICES"] = str(args.num_cores)
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(
            args.num_cores
        )
        try:
            print("STARTING SPAWN", flush=True)
            xmp.spawn(prep_and_train_fn, args=(args,), start_method="spawn")
        except Exception as e:
            print(f"Error spawning processes: {e}", flush=True)
            raise e


if __name__ == "__main__":
    print("Starting...")
    main()
