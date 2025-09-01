import argparse
import torch
import traceback
import sys

#import deepspeed
try:
    from torch_xla.runtime import world_size, global_ordinal
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    print("XLA import successful")

    # Make `torch.xla` point at the installed torch_xla package
    #import torch_xla
    #torch.xla = torch_xla
    #sys.modules["torch.xla"] = torch_xla
except ImportError as e:
    print(f"XLA import failed: {e}")
    exit(1)

from transformers import (
LlamaTokenizerFast,
LlamaConfig,
LlamaForCausalLM,
DataCollatorForLanguageModeling
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from datasets import load_dataset, DatasetDict, DatasetInfo
from safetensors.torch import load_file as load_safetensors

import wandb
import os
import io
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

# try:
#     os.environ.pop('TPU_PROCESS_ADDRESSES')
#     os.environ.pop('CLOUD_TPU_TASK_ID')
# except:
#     print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

print("TPU_NUM_DEVICES identified at system level:", os.environ.get("TPU_NUM_DEVICES"), flush=True)
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

# def tokenize_function(examples, tokenizer, max_seq_length):
#     # here you can actually add a chunker to split the text into smaller parts, of max_len
#     output= tokenizer(examples["text"],
#                     truncation=False,
#                     max_length=max_seq_length,
#                     padding="max_length",
#                     return_overflowing_tokens=True,
#                     return_length=True
#     )
#     input_batch = []
#     for length, input_ids in zip(output['length'], output['input_ids']):
#         input_batch.append({
#             "input_ids": input_ids,
#             "attention_mask": [1] * length
#         })
#     return {"input_ids": input_batch}

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

def load_sharded_dataset(datasets, dformat, args):
    # For single process mode, use process-based indices instead of TPU core indices
    if args.num_cores == 1:
        num_shards = 1
        shard_idx = 0
    else:
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
    # For streaming data, each process needs to load and tokenize independently
    # but we stagger the start times to avoid resource contention

    # Stagger the start times to avoid all processes hitting the storage simultaneously
    current_ordinal = global_ordinal()
    stagger_delay = current_ordinal * 2  # 2 seconds per core

    if stagger_delay > 0:
        print(f"Process {current_ordinal}: Waiting {stagger_delay}s to stagger dataset loading...")
        sleep(stagger_delay)

    # Load and tokenize dataset
    if args.pre_tokenized:
        datasets = {
            "train": args.dataset_dir + f"/train_{args.max_seq_length}.{args.dataset_format}",
            "validation": args.dataset_dir + f"/validation_{args.max_seq_length}.{args.dataset_format}"
        }
        if args.streaming_data:
            print(f"Process {current_ordinal}: Loading pre-tokenized streaming dataset...", flush=True)
            tokenized_dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=True,
                    keep_in_memory=False,
                ).shuffle(buffer_size=args.shuffle_buffer_size)
        else:
            print(f"Process {current_ordinal}: Loading pre-tokenized dataset...", flush=True)
            tokenized_dataset = load_dataset(
                args.dataset_format,
                data_files=datasets,
                streaming=False,
                keep_in_memory=True,
            )
    else:
        print(f"Process {current_ordinal}: Loading raw dataset for tokenization...", flush=True)
        print(f"Dataset location: {args.dataset_dir}", flush=True)

        train_loc = "train" if not args.debug else "validation"
        datasets = {"train": args.dataset_dir+f"/{train_loc}/*.{args.dataset_format}",
                    "validation": args.dataset_dir+f"/validation/*.{args.dataset_format}"}

        if args.streaming_data:
            try:
                if current_ordinal == 0:  # Only master process calculates this
                    ds_train_info = DatasetInfo.from_directory(args.dataset_dir+f"/{train_loc}/")
                    num_examples = ds_train_info['num_examples']
                    args.max_steps_per_epoch = num_examples // args.per_device_train_batch_size // max(world_size(), 1)
                    print(f"Maximum steps per epoch: {args.max_steps_per_epoch}", flush=True)
            except Exception as e:
                print(f"Could not obtain datasetinfo:{e}")
                pass

            print(f"Process {current_ordinal}: Init streaming dataset...", flush=True)
            dataset = load_dataset(
                    args.dataset_format,
                    data_files=datasets,
                    streaming=True,
                    keep_in_memory=False
                )
        else:
            print(f"Process {current_ordinal}: Init non-streaming dataset...", flush=True)
            if args.sharded_data:
                print("Sharding data...", flush=True)
                dataset = load_sharded_dataset(datasets, args.dataset_format, args)
            else:
                dataset = load_dataset(
                        args.dataset_format,
                        data_files=datasets,
                        streaming=False,
                        keep_in_memory=True
                    )

        # For streaming data, don't use multiprocessing (num_proc=1 or omit)
        # Each TPU core is already a separate process
        print(f"Process {current_ordinal}: Tokenizing dataset...", flush=True)
        tokenize_fn = partial(tokenize_function, tokenizer=args.tokenizer, max_seq_length=args.max_seq_length)

        # Remove num_proc for streaming to avoid conflicts
        if args.streaming_data:
            tokenized_dataset_raw = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["text", "id", "source", "approx_token_counts_translated", "approx_token_counts_original"]
            )
        else:
            tokenized_dataset_raw = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["text", "id", "source", "approx_token_counts_translated", "approx_token_counts_original"],
                num_proc=1  # Use single process to avoid conflicts
            )

        print(f"Process {current_ordinal}: Performing chunking tokenized data...", flush=True)
        group_fn = partial(group_texts, max_seq_length=args.max_seq_length, pad_token=args.tokenizer.pad_token_id)

        if args.streaming_data:
            tokenized_dataset = tokenized_dataset_raw.map(group_fn, batched=True)
        else:
            tokenized_dataset = tokenized_dataset_raw.map(
                group_fn,
                batched=True,
                num_proc=1,
                desc=f"Grouping texts in chunks of {args.max_seq_length}"
            )

        del tokenized_dataset_raw

    print(f"Process {current_ordinal}: Dataset preparation complete", flush=True)

    # Optional: Add a barrier to ensure all processes finish around the same time
    xm.rendezvous("dataset_prep_complete")

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
            import traceback
            traceback.print_exc()
            # Don't continue on XLA tensor errors - these need to be fixed
            if "XLA" in str(e) or "tensor" in str(e).lower():
                print("XLA/tensor error detected - stopping iteration")
                raise
            continue

def train_fn(tokenized_dataset, device, args):
    # Initialize wandb for the master process
    print(f"Process {global_ordinal()}: Starting train_fn...", flush=True)
    if global_ordinal() == 0: #.is_master_ordinal():
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="Llama 3.2 - 1B TPU CPT",
            config={
                "TPU ID": args.TPU_NAME,
                "TPU DISK": args.TPU_DISK,
                "learning_rate": args.learning_rate,
                "architecture": "Llama 3.2 - 1B",
                "dataset": args.dataset_dir,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.per_device_train_batch_size,
            },
            mode="online",
            dir="/home/bes3/temp"
        )
        wandb.run.log_code(root="/home/bes3/models", name="cpt_llama")

    # Load model configuration
    #config = RobertaConfig.from_pretrained(args.model_name)

    # Load pre-trained model
    print(f"Process {global_ordinal()}: Loading the LM...", flush=True)
    if isinstance(args.checkpoint_path, str) & (args.checkpoint_path != ""):
        print(f"Process {global_ordinal()}: Loading model from checkpoint: {args.checkpoint_path}", flush=True)

        #model_config = LlamaConfig.from_pretrained(args.model_name)
        config = LlamaConfig.from_pretrained(args.model_name,
            token=args.huggingface_token)

        # Create model with config (avoids meta parameters)
        with torch.device("cpu"):
            model = LlamaForCausalLM(config)
        print(f"Process {global_ordinal()}: Empty model created on CPU", flush=True)

        if args.checkpoint_path.startswith('gs://'):
            # Parse GCS path
            bucket_name = args.checkpoint_path.split('/')[2]
            blob_name = '/'.join(args.checkpoint_path.split('/')[3:])
            local_path = f'/tmp/checkpoint.{args.checkpoint_path.split(".")[-1]}'  # Temporary local path to store the downloaded file
            checkpoint = load_from_gcs(bucket_name,
                                       blob_name,
                                       local_path,
                                       device)
            sleep(1)
            print(f"Process {global_ordinal()}: Checkpoint downloaded...", flush=True)
            xm.rendezvous("checkpoint_downloaded")
        else:
            if args.checkpoint_path.endswith('.safetensors'):
                checkpoint = load_safetensors(args.checkpoint_path, device=device)
            else:
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
            sleep(1)
            print(f"Process {global_ordinal()}: Checkpoint downloaded...", flush=True)
            xm.rendezvous("checkpoint_downloaded")


        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        sleep(1)
        print(f"Process {global_ordinal()}: Checkpoint loaded in memory..", flush=True)
    else:
        with torch.device("cpu"):
            model = LlamaForCausalLM.from_pretrained(args.model_name,
                token=args.huggingface_token,
                device_map=None,
                _fast_init=False,
                low_cpu_mem_usage=False
            )

    xm.rendezvous("checkpoint_loaded_in_memory")

    # Debug single core mode
    print(f"Process {global_ordinal()}: world_size={world_size()}, global_ordinal={global_ordinal()}, num_cores={args.num_cores}", flush=True)

    # For single core mode, simplify the model loading
    if args.num_cores == 1:
        print(f"Process {global_ordinal()}: Single core mode - loading model to TPU device {device}...", flush=True)
        try:
            model = model.to(device=device, dtype=torch.bfloat16)
            xm.mark_step()  # Ensure model transfer completes
            print(f"Process {global_ordinal()}: Model loaded to device successfully", flush=True)
            print(f"Process {global_ordinal()}: Model device after loading: {next(model.parameters()).device}", flush=True)
        except Exception as e:
            print(f"Process {global_ordinal()}: Error loading model to device: {e}", flush=True)
            raise
    else:
        # Original multi-core logic
        for i in range(world_size()):
            if global_ordinal() == i:
                print(f"Process {global_ordinal()}: My turn to load model to TPU device {device}...", flush=True)
                try:
                    model = model.to(device=device, dtype=torch.bfloat16)
                    # model.gradient_checkpointing_enable()
                    print(f"Process {global_ordinal()}: Checkpoint loaded on device successfully", flush=True)
                except Exception as e:
                    print(f"Process {global_ordinal()}: Error loading model to device: {e}", flush=True)
                    raise

            # Wait for current process to finish before next one starts
            xm.rendezvous(f"model_to_device_{i}")

    print(f"Process {global_ordinal()}: All models loaded to devices", flush=True)

    # print("After checkpointing, model on device:", next(model.parameters()).device, flush=True)
    # if hasattr(model, 'tie_weights'):
    #     model.tie_weights()
    # Set up data collator
    print(f"Process: {global_ordinal()}. Setting up data collator.")
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,
                                                    mlm=False)
    xm.rendezvous("datacollator")

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

        # For single process mode, use process-based indices instead of TPU core indices
        if args.num_cores == 1:
            num_shards = 1
            shard_id = 0
        else:
            num_shards = world_size()  # Total number of TPU cores (or processes)
            shard_id = global_ordinal()  # Unique id for the current process
        print(f"Process {global_ordinal()}: Num shards: {num_shards}, shard id: {shard_id}", flush=True)

        try:
            sharded_train_dataset = tokenized_dataset["train"].shard(num_shards=num_shards, index=shard_id)
            print(f"Process {global_ordinal()}: Sharded dataset created successfully", flush=True)
        except Exception as e:
            print(f"Process {global_ordinal()}: Error creating sharded dataset: {e}", flush=True)
            raise

        print(f"Process {global_ordinal()}: Creating streaming DataLoader...", flush=True)

        train_dataloader = torch.utils.data.DataLoader(
            sharded_train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=False,
            drop_last=True
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

    # For single core, try alternative data loading approach
    if args.num_cores == 1:
        print(f"Process {global_ordinal()}: Using single core data loading", flush=True)
        # For single core, we might not need MpDeviceLoader
        xla_train_loader = train_dataloader
        xla_validation_loader = validation_dataloader
    else:
        xla_train_loader = pl.MpDeviceLoader(train_dataloader, xm.xla_device())
        xla_validation_loader = pl.MpDeviceLoader(validation_dataloader, xm.xla_device())

    print(f"XLA device is: {xm.xla_device()}", flush=True)
    print(f"Device for training: {device}", flush=True)

    xm.rendezvous("data_loader")

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #optimizer = torch.optim.Adafactor(model.parameters(), lr=args.learning_rate, #weight_decay=args.weight_decay, scale_parameter=False, relative_step=False)

    steps_per_epoch = args.max_steps_per_epoch if args.streaming_data else len(train_dataloader)
    save_steps = int(steps_per_epoch * args.save_epoch_percentage) if args.save_epoch_percentage<1 else args.save_epoch_percentage
    total_steps = steps_per_epoch * args.num_train_epochs
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)

    if args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=args.num_cycles)

    # ds_config = {
    #     "zero_optimization": {
    #         "stage": 3,
    #         "offload_param": {
    #             "device": "cpu"
    #         }
    #     },
    #     "bf16": {
    #         "enabled": True
    #     }
    # }

    # # Replace model, optimizer, scheduler with a DeepSpeed engine
    # model, optimizer, _, scheduler = deepspeed.initialize(
    #     model=model,
    #     optimizer=optimizer,
    #     lr_scheduler=scheduler,
    #     config=ds_config
    # )

    xm.rendezvous("start_training")
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
        xm.rendezvous(f"start_epoch_{epoch}")
        print(f"Starting with epoch {epoch}...for process {global_ordinal()}", flush=True)

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
                # Debug tensor information for first few batches
                if step < 3:
                    print(f"Process {global_ordinal()}: Processing batch {step}...", flush=True)

                if step == 0:
                    print(f"Process {global_ordinal()}: First batch tensor info:", flush=True)
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            is_xla = 'xla' in str(v.device).lower()
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, is_xla={is_xla}", flush=True)

                    # Check model device
                    model_device = next(model.parameters()).device
                    print(f"  Model parameters device: {model_device}", flush=True)
                    print(f"  Target device: {device}", flush=True)

                    # Verify XLA is working
                    try:
                        test_tensor = torch.tensor([1.0]).to(device)
                        print(f"  Test tensor device: {test_tensor.device}, is_xla: {'xla' in str(test_tensor.device).lower()}", flush=True)
                    except Exception as tensor_e:
                        print(f"  Error creating test tensor: {tensor_e}", flush=True)

                    # Verify model embeddings are on XLA device
                    try:
                        embedding_device = model.model.embed_tokens.weight.device
                        print(f"  Model embeddings device: {embedding_device}, is_xla: {'xla' in str(embedding_device).lower()}", flush=True)

                        # For single core, ensure model is properly on XLA device
                        if args.num_cores == 1 and 'xla' not in str(embedding_device).lower():
                            print(f"  WARNING: Model not on XLA device in single core mode, forcing move...", flush=True)
                            model = model.to(device)
                            xm.mark_step()  # Ensure transfer completes
                            new_embedding_device = model.model.embed_tokens.weight.device
                            print(f"  Model moved - new embeddings device: {new_embedding_device}", flush=True)
                    except AttributeError as ae:
                        print(f"  Could not access model embeddings: {ae}", flush=True)

                # Handle tensor placement - simplified approach for single core
                if args.num_cores == 1:
                    # For single core, be more explicit about tensor movement
                    new_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            # Ensure tensor is on XLA device
                            if not ('xla' in str(v.device).lower()):
                                if v.dtype == torch.float32:
                                    new_batch[k] = v.to(device, dtype=torch.bfloat16)
                                else:
                                    new_batch[k] = v.to(device)
                            else:
                                # Already on XLA device, just convert dtype if needed
                                if v.dtype == torch.float32:
                                    new_batch[k] = v.to(dtype=torch.bfloat16)
                                else:
                                    new_batch[k] = v
                        else:
                            new_batch[k] = v
                    batch = new_batch

                    # For first few batches, log tensor devices after processing
                    if step < 3:
                        print(f"  Batch {step} - Final tensor devices:", flush=True)
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                print(f"    {k}: {v.device} (shape: {v.shape})", flush=True)

                        # Ensure XLA operations are synchronized before model forward pass
                        xm.mark_step()
                else:
                    # Original logic for multi-core
                    batch = {k: v.to(device=device, dtype=torch.bfloat16) if v.dtype==torch.float32 else v.to(device) for k, v in batch.items()}

                samples_seen_global = (step+1) * args.per_device_train_batch_size * world_size()
                loss = model(**batch).loss / args.gradient_accumulation_steps
                nan = torch.isnan(loss)
                if nan:
                    optimizer.zero_grad(set_to_none=True)
                    print(f"[Rank {global_ordinal()}] NaN loss; skipping backward but keeping sync", flush=True)
                    loss = torch.nan_to_num(loss, nan=0.0)

                loss.backward()

                total_loss += loss.item()*args.gradient_accumulation_steps
                sub_total_loss += loss.item()*args.gradient_accumulation_steps

                if  (step+1) % args.gradient_accumulation_steps == 0:
                    #model.step()              # does optimizer_step + zero_grad via DeepSpeed
                    #print(f"[Rank {global_ordinal()}] Performing optimization", flush=True)
                    # Add before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if grad_norm > args.max_grad_norm:
                        print(f"[Rank {global_ordinal()}] Gradient norm {grad_norm:.2f} was clipped to {args.max_grad_norm} at step {step}")
                    # reactivate grad norm later
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    xm.optimizer_step(optimizer, barrier=True)
                    scheduler.step()
                    xm.mark_step()
                    optimizer.zero_grad()
                    #xm.clear_all_gradients()

                    total_step += args.gradient_accumulation_steps

                #xm.mark_step()
                    #xm.clear_all_gradients()       # drops stale XLA gradient buffers

                #if (step+1) %  (args.gradient_accumulation_steps - 1)== 0:
                    #mem = xm.get_memory_info(xm.xla_device())
                    #print(f"[Rank {global_ordinal()}] Free HBM: {','.join([str(round(m/1e6,3)) for m in mem.values()])} GB")

                if (step+1) % args.logging_steps == 0: # xm.is_master_ordinal():
                    local_avg_loss = total_loss / (step + 1)
                    local_avg_loss_N = sub_total_loss / max(sub_step, 1)

                    global_avg_loss = xm.mesh_reduce("loss", local_avg_loss, np.mean)
                    global_avg_loss_N = xm.mesh_reduce("loss_N", local_avg_loss_N, np.mean)

                    perplexity = math.exp(global_avg_loss)
                    perplexity_N = math.exp(global_avg_loss_N)

                    sub_step = 0
                    sub_total_loss = 0.

                    xm.rendezvous("before_wandb_log")
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
                        }, commit = False)
                    xm.rendezvous("after_wandb_log")
                if args.debug:
                    break

                #total_step += 1
                sub_step +=1

                if (samples_seen_global % save_steps == 0):
                    xm.rendezvous("before_model_saving")
                    if (global_ordinal() ==0):
                        print("Saving model...", flush=True)

                        wandb.log({"epoch": epoch, "step": samples_seen_global}, commit=True)
                        wandb.finish()

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
                                subprocess.run(["gsutil", "mv", os.path.join(args.output_dir, local_output_dir), new_gcs_path], check=True)

                        else:
                            model_cpu = copy.deepcopy(model).to("cpu")
                            model_cpu.save_pretrained(args.output_dir, save_serialization=True)
                    xm.rendezvous("after_model_saving")
                else:
                    pass

            except Exception as e:
                print(f"Error occurred during processing batch {step}: {e}", flush=True)
                print("Full traceback:")
                traceback.print_exc()
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
            }, commit=True)

        # Save model checkpoint
        if global_ordinal() == 0:
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
        else:
            pass

    # Finish wandb run
    if global_ordinal() == 0:
        wandb.finish()

def prep_and_train_fn(index, args):
    device = xm.xla_device()

    # Verify all cores are participating
    world_size_val = world_size()
    ordinal = global_ordinal()

    print(f"Process {index}: ordinal={ordinal}, world_size={world_size_val}, device={device}", flush=True)

    xm.rendezvous("prep_start")
    tokenized_dataset = prep_fn(args)

    xm.rendezvous("train_start")
    train_fn(tokenized_dataset, device, args)

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
    xm.mark_step()  # Ensure mesh_reduce operation is executed

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
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pre_tokenized", action='store_true', default=False)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_schedule", type=str, choices=["linear", "cosine"], default="linear", help="linear=warmup→decay; oscillatory=cosine‐restarts")
    parser.add_argument("--num_cycles", type=float, default=10, help="number of cycles for oscillatory schedule (defaults to 10)")
    parser.add_argument("--eta_min", type=float, default=1e-6,
                        help="minimum LR for oscillatory schedule")
    parser.add_argument("--max_grad_norm", type=float, default=14.4)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--logging_steps", type=int, default=1_000)
    parser.add_argument("--save_epoch_percentage", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cores", type=int, default=4)
    parser.add_argument("--keep_in_memory", action='store_true')
    parser.add_argument("--streaming_data", action='store_true')
    parser.add_argument("--sharded_data", action='store_true')
    parser.add_argument("--max_steps_per_epoch", type=int, default=50_000_000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=1_000)
    parser.add_argument("--shuffle_dataset_path", type=str, default="/home/bob/tmp/shuffle.parquet")
    parser.add_argument("--shuffle_dataset_ext", type=str, default=None)
    parser.add_argument("--shuffle_dataset", action='store_true')
    parser.add_argument("--shuffle_force_update", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl")
    parser.add_argument("--wandb_key", type=str, required=True,help="Weights & Biases API key")
    parser.add_argument("--huggingface_token", type=str, required=True)
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


    args.tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer_name_or_path, token=args.huggingface_token)
    args.tokenizer.model_max_length = args.max_seq_length
    args.tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if args.shuffle_dataset:
            args.dataset_dir = args.shuffle_dataset_path
            args.dataset_format = 'arrow'
            print("Continuing with Arrow format", flush=True)

    sleep(5)

    # Spawn processes
    print(f"Initializing TPU...with {args.num_cores} cores")
    print("Spawning processes...")

    if args.num_cores == 1:
        print("Running single process...", flush=True)
        prep_and_train_fn(0, args)
    else:
        # set TPU_NUM_DEVICES to args.num_cores
        os.environ['TPU_NUM_DEVICES'] = str(args.num_cores)
        # Ensure XLA uses all available cores
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=' + str(args.num_cores)
        try:
            xmp.spawn(prep_and_train_fn, args=(args,), start_method='spawn')
        except Exception as e:
            print(f"Error spawning processes: {e}", flush=True)
            raise e
if __name__ == "__main__":
    print("Starting...")
    main()
