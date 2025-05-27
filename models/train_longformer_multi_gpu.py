"""
This is the main script to continue pre-training a Longformer model on GPU.
"""
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LongformerTokenizerFast,  # Use Longformer tokenizer
    LongformerConfig,  # Use Longformer config
    LongformerForMaskedLM,  # Use Longformer model
    DataCollatorForLanguageModeling
)
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
from functools import partial
from itertools import chain

from time import sleep

# Import LazyGroupingDataset with GPU compatibility
class LazyGroupingDataset(torch.utils.data.IterableDataset):
    """
    A wrapper dataset that applies grouping lazily during iteration.
    This allows for on-the-fly text grouping during training rather than preprocessing the entire dataset upfront.

    For streaming datasets, this is particularly beneficial as it:
    1. Reduces startup time dramatically
    2. Processes only the data needed for each batch
    3. Maintains the streaming nature of the dataset
    """

    def __init__(
        self,
        dataset,
        max_seq_length: int,
        pad_token: int = 0,
        batch_size: int = 8,
        streaming: bool = True
    ):
        """
        Initialize the lazy grouping dataset wrapper.

        Args:
            dataset: The underlying dataset containing tokenized examples
            max_seq_length: Maximum sequence length for grouped examples
            pad_token: Token ID to use for padding (default: 0)
            batch_size: Batch size for grouping
            streaming: Whether the dataset is streaming (iterable) or not
        """
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.batch_size = batch_size
        self.streaming = streaming

    def __iter__(self):
        """
        Iterate through the dataset, applying grouping on-the-fly.

        Returns:
            Iterator yielding batches of grouped examples
        """
        # Create an iterator from the underlying dataset
        dataset_iter = iter(self.dataset)

        # Keep yielding batches until we're done
        while True:
            try:
                # Get a batch of examples
                batch = []
                for _ in range(self.batch_size):
                    try:
                        item = next(dataset_iter)
                        batch.append(item)
                    except StopIteration:
                        if not batch:  # If batch is empty, we're truly done
                            raise
                        break  # Otherwise process the partial batch

                if not batch:
                    break

                # Convert list of dicts to dict of lists
                batch_dict = {k: [item[k] for item in batch] for k in batch[0].keys()}

                # Apply grouping to this batch
                grouped_batch = self._group_texts(batch_dict)

                # Yield each example in the grouped batch
                for i in range(len(grouped_batch[list(grouped_batch.keys())[0]])):
                    # Return the example without converting to tensor
                    # Let the data collator handle the tensor conversion
                    example = {k: grouped_batch[k][i] for k in grouped_batch.keys()}
                    yield example

            except StopIteration:
                # No more data
                break

    def _group_texts(self, examples):
        """
        Group already tokenized texts into chunks of max_seq_length while respecting sample boundaries.

        Args:
            examples: Dictionary with keys like 'input_ids', 'attention_mask', etc. where each value
                     is a list of tokenized examples

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
            num_chunks = (example_length + self.max_seq_length - 1) // self.max_seq_length  # Ceiling division

            # Split each feature into chunks
            for k, tokens in current_example.items():
                # Create chunks of max_seq_length
                chunks = []
                for j in range(0, example_length, self.max_seq_length):
                    chunk = tokens[j:min(j + self.max_seq_length, example_length)]

                    # Pad if necessary
                    if len(chunk) < self.max_seq_length:
                        chunk = chunk + [self.pad_token] * (self.max_seq_length - len(chunk))

                    chunks.append(chunk)

                # If we don't have enough chunks (unlikely but possible with different length features)
                while len(chunks) < num_chunks:
                    chunks.append([self.pad_token] * self.max_seq_length)

                # Add the chunks to the result
                result[k].extend(chunks)

        return result

    def __len__(self):
        """
        Return the length of the dataset if available.
        For streaming datasets, this may not be available.

        Returns:
            Length of the dataset if available, else None
        """
        if not self.streaming and hasattr(self.dataset, "__len__"):
            # This is a very rough estimate as grouping will change the actual length
            return len(self.dataset) * self.max_seq_length
        return None


def get_rank():
    """Return the rank of the current process in DDP."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Return the number of processes in DDP."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


def setup_ddp(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training environment."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_from_gcs(bucket_name, blob_name, local_path, device):
    """Load a file from Google Cloud Storage."""
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return torch.load(local_path, map_location=device)


def safe_iter(data_loader):
    """Iterate over a dataloader safely."""
    while True:
        try:
            yield next(data_loader)
        except StopIteration:
            break
        except Exception as e:
            print(f"Error in data loading: {e}", flush=True)
            continue


def shuffle_and_save_dataset(dataset, output_dir):
    """Shuffle a dataset and save it to disk."""
    # Shuffle each split
    shuffled_dataset = DatasetDict()
    for key, ds in dataset.items():
        shuffled_dataset[key] = ds.shuffle(seed=42)
    
    # Save to disk
    shuffled_dataset.save_to_disk(output_dir)
    
    return shuffled_dataset


class ShardedShuffleDataset(torch.utils.data.IterableDataset):
    """Dataset wrapper that shards and shuffles streaming datasets."""
    
    def __init__(self, dataset, buffer_size, num_shards=1, shard_id=0):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.num_shards = num_shards
        self.shard_id = shard_id
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # If DataLoader workers are enabled, further shard the dataset
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            effective_shard_id = self.shard_id * num_workers + worker_id
            effective_num_shards = self.num_shards * num_workers
        else:
            effective_shard_id = self.shard_id
            effective_num_shards = self.num_shards
        
        # Apply sharding to the dataset
        sharded_dataset = self.dataset.shard(num_shards=effective_num_shards, index=effective_shard_id)
        
        # Apply shuffling with a buffer
        shuffled_dataset = sharded_dataset.shuffle(buffer_size=self.buffer_size)
        
        yield from shuffled_dataset


def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
    )


def load_from_gcs(bucket_name, blob_name, local_path, device):
    """Load a file from Google Cloud Storage."""
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return torch.load(local_path, map_location=device)


def load_sharded_dataset(data_files, dataset_format):
    """Load a dataset with sharding."""
    datasets = DatasetDict()
    for split, pattern in data_files.items():
        datasets[split] = load_dataset(
            dataset_format,
            data_files=pattern,
            split='train'
        )
    return datasets


def group_texts(examples, max_seq_length, pad_token):
    """Group texts into chunks of max_seq_length."""
    # Concatenate all texts
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Drop small remainder if necessary
    total_length = (total_length // max_seq_length) * max_seq_length
    
    # Split by chunks of max_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    
    # Add padding to shorter sequences if needed
    for k in result.keys():
        for i in range(len(result[k])):
            if len(result[k][i]) < max_seq_length:
                result[k][i] = result[k][i] + [pad_token] * (max_seq_length - len(result[k][i]))
    
    return result


def prep_fn(args, rank):
    """Prepare the dataset for training."""
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
                args.max_steps_per_epoch = num_examples // args.per_device_train_batch_size // max(get_world_size(), 1)
            except Exception as e:
                print(f"Could not obtain datasetinfo:{e}")
                pass

            print("Init streaming dataset...", flush=True)
            dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=True,
                    keep_in_memory=False
                )
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

        opt_kwargs = {'num_proc': args.num_workers} if args.streaming_data==False else {}
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


def train_fn(tokenized_dataset, args, rank, gpu_id):
    """Train the model."""
    device = torch.device(f"cuda:{gpu_id}")
    
    # Initialize wandb for the master process
    if is_main_process():
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="Longformer GPU pretraining from scratch",
            config={
                "learning_rate": args.learning_rate,
                "architecture": "Longformer",
                "dataset": args.dataset_dir,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
                "lazy_grouping": args.lazy_grouping,
                "num_gpus": args.num_gpus
            },
            mode="online",
            dir="/tmp/wandb"
        )

    # Load model configuration
    if is_main_process():
        print("Loading the LM ...", flush=True)
    model_config = LongformerConfig.from_pretrained(args.model_name)
    model_config.bos_token_id = args.tokenizer.bos_token_id
    model_config.eos_token_id = args.tokenizer.eos_token_id
    model_config.pad_token_id = args.tokenizer.pad_token_id
    model_config.cls_token_id = args.tokenizer.cls_token_id
    model_config.sep_token_id = args.tokenizer.sep_token_id
    model_config.vocab_size = args.tokenizer.vocab_size
    model_config.max_position_embeddings = args.max_seq_length+2

    model = LongformerForMaskedLM(model_config)

    if isinstance(args.checkpoint_path, str) and (args.checkpoint_path != ""):
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

    model = model.to(device)
    
    # Use mixed precision with autocast
    fp16 = True if args.precision == "fp16" else False
    bf16 = True if args.precision == "bf16" else False
    
    # Wrap model with DDP when using multiple GPUs
    if args.num_gpus > 1:
        model = DDP(model, device_ids=[gpu_id])

    # Set up data collator
    if is_main_process():
        print("Setting up data collator...", flush=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,
                                                    mlm=True,
                                                    mlm_probability=args.mlm_probability)

    # Create dataloaders
    if args.streaming_data:
        if is_main_process():
            print("Instantiate sharded dataset...", flush=True)
        num_shards = get_world_size()  # Total number of processes
        shard_id = get_rank()  # Unique id for the current process
        print(f"Num shards: {num_shards}, shard id: {shard_id}", flush=True)
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

        if is_main_process():
            print("Starting the streaming DataLoader...", flush=True)

        train_dataloader = torch.utils.data.DataLoader(
            _sharded_train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=args.dataloader_num_workers
        )
    else:
        if is_main_process():
            print("Starting the DistributedSampler...", flush=True)
        train_sampler = DistributedSampler(
            tokenized_dataset["train"],
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        )
        # Create dataloaders
        if is_main_process():
            print("Starting the DataLoader...", flush=True)
        train_dataloader = torch.utils.data.DataLoader(
            tokenized_dataset["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            sampler=train_sampler,
            num_workers=args.dataloader_num_workers,
            pin_memory=True
        )
    
    # For validation, we can use a single process
    if is_main_process():
        validation_dataloader = torch.utils.data.DataLoader(
            tokenized_dataset["validation"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=True
        )
    
    if not args.streaming_data:
        del tokenized_dataset
        gc.collect()

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_steps_per_epoch if args.streaming_data else len(train_dataloader)
    save_steps = args.save_epoch_percentage if args.save_epoch_percentage>1. else int(steps_per_epoch * args.save_epoch_percentage)
    total_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)

    if is_main_process():
        print("Starting training...", flush=True)
        print(f"Total steps: {total_steps}", flush=True)
        print(f"Total epochs: {args.num_train_epochs}", flush=True)
        print(f"Total warmup steps: {args.num_warmup_steps}", flush=True)

    # Training loop
    total_step = 0
    miss_steps = 0
    print(f"ENTERING THE TRAINING LOOP with: {args.num_train_epochs} epochs", flush=True)
    
    # Scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    
    for epoch in range(args.num_train_epochs):
        print(f"Starting with epoch {epoch}...", flush=True)
        total_loss = 0.
        sub_total_loss = 0.
        sub_step = 0
        model.train()
        
        if not args.streaming_data:
            train_sampler.set_epoch(epoch)
        
        print(f"Entering data loader iterator for epoch {epoch}...", flush=True)
        step = -1
        
        for step, batch in enumerate(safe_iter(train_dataloader)):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Mixed precision training
                with torch.cuda.amp.autocast(enabled=fp16 or bf16, dtype=torch.bfloat16 if bf16 else torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # Scale loss and perform backward pass
                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                sub_total_loss += loss.item()
                
                if (step+1) % args.gradient_accumulation_steps == 0:
                    if fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    total_step += args.gradient_accumulation_steps
                
                if (step+1) % args.logging_steps == 0:
                    # Synchronize for logging
                    if args.num_gpus > 1:
                        dist.barrier()
                    
                    local_avg_loss = total_loss / (step + 1)
                    local_avg_loss_N = sub_total_loss / (sub_step + 1)
                    
                    # Collect and average losses across all processes
                    if args.num_gpus > 1:
                        loss_tensor = torch.tensor([local_avg_loss, local_avg_loss_N], device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        global_avg_loss = loss_tensor[0] / get_world_size()
                        global_avg_loss_N = loss_tensor[1] / get_world_size()
                    else:
                        global_avg_loss = local_avg_loss
                        global_avg_loss_N = local_avg_loss_N
                    
                    perplexity = math.exp(global_avg_loss)
                    perplexity_N = math.exp(global_avg_loss_N)
                    
                    sub_step = 0
                    sub_total_loss = 0.
                    
                    if is_main_process():
                        print(f"Logging for epoch: {epoch}, step: {step}, train_perplexity_N:{perplexity_N}", flush=True)
                        wandb.log({
                            "train_global_average_loss": global_avg_loss,
                            "train_global_average_loss_N": global_avg_loss_N,
                            "train_perplexity": perplexity,
                            "train_perplexity_N": perplexity_N,
                            "epoch": epoch,
                            "step": step,
                            "total_step": total_step,
                            "learning_rate": scheduler.get_last_lr()[0]
                        })
                
                if args.debug:
                    break
                
                total_step += 1
                sub_step += 1
                
                # Save checkpoint periodically
                if is_main_process() and ((step+1) % save_steps == 0):
                    if args.num_gpus > 1:
                        dist.barrier()
                    
                    print("Saving model...", flush=True)
                    if args.output_dir.startswith("gs://"):
                        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                            local_output_dir = tmpdirname
                            print(f"Saving model to {local_output_dir}...", flush=True)
                            
                            # Get the model state dict
                            if args.num_gpus > 1:
                                model_to_save = model.module
                            else:
                                model_to_save = model
                            
                            model_to_save.save_pretrained(local_output_dir, safe_serialization=True)
                            
                            print(f"Uploading model to {args.output_dir}...", flush=True)
                            subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
                            
                            # rename the directory on GCS to the model name, date, and epoch_step
                            new_dir_name = f"{args.model_name}_latest"
                            new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                            subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir, "*"), new_gcs_path], check=True)
                    else:
                        # Get the model state dict
                        if args.num_gpus > 1:
                            model_to_save = model.module
                        else:
                            model_to_save = model
                        
                        model_to_save.save_pretrained(args.output_dir, safe_serialization=True)
                    
                    if args.num_gpus > 1:
                        dist.barrier()
            
            except Exception as e:
                print(f"Error occurred during processing batch {step}: {e}", flush=True)
                miss_steps += 1
                continue
            
            if (step+1) % steps_per_epoch == 0:
                break
        
        # Evaluate on validation set (only on main process)
        if is_main_process():
            val_loss = evaluate(model, validation_dataloader, device, fp16, bf16, args.num_gpus > 1)
            val_ppl = math.exp(val_loss)
            
            wandb.log({
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "epoch": epoch,
                "miss_steps": miss_steps
            })
        
        # Synchronize processes
        if args.num_gpus > 1:
            dist.barrier()
        
        # Save model checkpoint at the end of epoch
        if is_main_process():
            print("Saving end-of-epoch model...", flush=True)
            if args.output_dir.startswith("gs://"):
                with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdirname:
                    local_output_dir = tmpdirname
                    print(f"Saving model to {local_output_dir}...", flush=True)
                    
                    # Get the model state dict
                    if args.num_gpus > 1:
                        model_to_save = model.module
                    else:
                        model_to_save = model
                    
                    model_to_save.save_pretrained(local_output_dir, safe_serialization=True)
                    
                    print(f"Uploading model to {args.output_dir}...", flush=True)
                    subprocess.run(["gsutil", "-m", "cp", "-r", local_output_dir, args.output_dir], check=True)
                    
                    # rename the directory on GCS
                    current_time = datetime.datetime.now().strftime("%Y%m%d")
                    new_dir_name = f"{args.model_name}_epoch{epoch}_{current_time}"
                    new_gcs_path = os.path.join(args.output_dir, new_dir_name)
                    # Rename the directory on GCS
                    subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir), new_gcs_path], check=True)
            else:
                if args.num_gpus > 1:
                    model_to_save = model.module
                else:
                    model_to_save = model
                
                model_to_save.save_pretrained(args.output_dir, safe_serialization=True)
        
        # Synchronize processes after saving
        if args.num_gpus > 1:
            dist.barrier()
    
    # Finish wandb run
    if is_main_process():
        wandb.finish()


def evaluate(model, eval_dataloader, device, fp16=False, bf16=False, is_ddp=False):
    """Evaluate the model on the validation dataset."""
    model.eval()
    total_loss = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=fp16 or bf16, dtype=torch.bfloat16 if bf16 else torch.float16):
                outputs = model(**batch)
            
            loss = outputs.loss
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    
    avg_loss = total_loss / total_examples
    
    # If using DDP, gather loss from all processes
    if is_ddp:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / get_world_size()
    
    return avg_loss


def prep_and_train_fn(rank, args):
    """Prepare and train the model for a single GPU."""
    # Set up device
    gpu_id = rank
    setup_ddp(rank, args.num_gpus)
    
    device = torch.device(f"cuda:{gpu_id}")
    
    # Prepare dataset
    tokenized_dataset = prep_fn(args, rank)
    
    # Train model
    train_fn(tokenized_dataset, args, rank, gpu_id)
    
    # Clean up
    cleanup_ddp()


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
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--keep_in_memory", action='store_true')
    parser.add_argument("--streaming_data", action='store_true')
    parser.add_argument("--sharded_data", action='store_true')
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--shuffle_dataset_path", type=str, default="/tmp/shuffle.parquet")
    parser.add_argument("--shuffle_dataset_ext", type=str, default=None)
    parser.add_argument("--shuffle_dataset", action='store_true')
    parser.add_argument("--shuffle_force_update", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument("--lazy_grouping", action='store_true', help="Use lazy grouping to process data on-the-fly during training (incompatible with --pre_tokenized)")
    parser.add_argument("--wandb_key", type=str, required=True,help="Weights & Biases API key")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for preprocessing")
    args = parser.parse_args()

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
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.num_gpus > 1:
        # Set cuda device for main process
        torch.cuda.set_device(0)
    
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

    # Launch training
    if args.num_gpus > 1:
        print(f"Launching training with {args.num_gpus} GPUs")
        torch.multiprocessing.spawn(
            prep_and_train_fn,
            args=(args,),
            nprocs=args.num_gpus
        )
    else:
        print("Running with a single GPU")
        prep_and_train_fn(0, args)


if __name__ == "__main__":
    main()