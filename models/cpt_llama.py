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

# import deepspeed
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.runtime import global_ordinal, world_size

    print("XLA import successful")

    # Make `torch.xla` point at the installed torch_xla package
    # import torch_xla
    # torch.xla = torch_xla
    # sys.modules["torch.xla"] = torch_xla
except ImportError as e:
    print(f"XLA import failed: {e}")
    exit(1)

import datetime
import gc
import io
import math
import random
import shutil
import subprocess
import threading
from functools import partial
from itertools import chain
from time import sleep

import fsspec
import numpy as np
import wandb
from datasets import DatasetDict, DatasetInfo, interleave_datasets, load_dataset
from gcsfs import GCSFileSystem
from google.cloud import storage
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# try:
#     os.environ.pop('TPU_PROCESS_ADDRESSES')
#     os.environ.pop('CLOUD_TPU_TASK_ID')
# except:
#     print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

print(
    "TPU_NUM_DEVICES identified at system level:",
    os.environ.get("TPU_NUM_DEVICES"),
    flush=True,
)
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


def tokenize_function(examples, tokenizer, max_seq_length, skip_eos=False):
    # here you can actually add a chunker to split the text into smaller parts, of max_len
    # Tokenize without padding first to properly add EOS tokens
    tokenized = tokenizer(
        examples["text"], truncation=False, padding=False, return_attention_mask=True
    )

    # Add EOS token to the end of each sequence (skip for pre-tokenized data)
    if not skip_eos:
        for i in range(len(tokenized["input_ids"])):
            # Check if EOS token is already present at the end
            if (
                len(tokenized["input_ids"][i]) == 0
                or tokenized["input_ids"][i][-1] != tokenizer.eos_token_id
            ):
                # Add EOS token at the end (don't replace, append)
                tokenized["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized["attention_mask"][i].append(1)

    # Truncate if sequence is too long (keeping EOS token for non-skip cases)
    for i in range(len(tokenized["input_ids"])):
        if len(tokenized["input_ids"][i]) > max_seq_length:
            if not skip_eos:
                tokenized["input_ids"][i] = tokenized["input_ids"][i][
                    : max_seq_length - 1
                ] + [tokenizer.eos_token_id]
                tokenized["attention_mask"][i] = tokenized["attention_mask"][i][
                    : max_seq_length - 1
                ] + [1]
            else:
                tokenized["input_ids"][i] = tokenized["input_ids"][i][:max_seq_length]
                tokenized["attention_mask"][i] = tokenized["attention_mask"][i][
                    :max_seq_length
                ]

    # Now pad all sequences to max_length manually since tokenizer.pad might not be available
    for i in range(len(tokenized["input_ids"])):
        current_length = len(tokenized["input_ids"][i])
        if current_length < max_seq_length:
            # Pad input_ids with pad_token_id
            padding_length = max_seq_length - current_length
            tokenized["input_ids"][i].extend([tokenizer.pad_token_id] * padding_length)
            # Pad attention_mask with 0s
            tokenized["attention_mask"][i].extend([0] * padding_length)

    return tokenized


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


def group_texts(examples, max_seq_length, pad_token=0, eos_token_id=None):
    """
    Group already tokenized texts into chunks of max_seq_length while preserving EOS tokens at document boundaries.

    Args:
        examples: Dictionary with keys like 'input_ids', 'attention_mask', etc. where each value
                 is a list of tokenized examples
        max_seq_length: Maximum sequence length
        pad_token: Token to use for padding (default: 0)
        eos_token_id: EOS token ID to preserve at document boundaries

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

        # If the sequence fits in one chunk, just pad it
        if example_length <= max_seq_length:
            for k, tokens in current_example.items():
                chunk = tokens[:]  # Make a copy
                # Pad if necessary
                if len(chunk) < max_seq_length:
                    pad_value = pad_token if k == "input_ids" else 0
                    chunk = chunk + [pad_value] * (max_seq_length - len(chunk))
                result[k].append(chunk)
        else:
            # Need to split into multiple chunks
            # For input_ids, preserve EOS token at the end of the last chunk
            input_ids = current_example["input_ids"]
            has_eos_at_end = (
                eos_token_id is not None
                and len(input_ids) > 0
                and input_ids[-1] == eos_token_id
            )

            # Split each feature into chunks
            for k, tokens in current_example.items():
                chunks = []

                # Create chunks of max_seq_length
                for j in range(0, example_length, max_seq_length):
                    chunk = tokens[j : min(j + max_seq_length, example_length)]

                    # For the last chunk of input_ids, ensure EOS token is preserved
                    if (
                        k == "input_ids"
                        and has_eos_at_end
                        and j + max_seq_length >= example_length
                    ):
                        # This is the last chunk, make sure it ends with EOS
                        if len(chunk) > 0 and chunk[-1] != eos_token_id:
                            if len(chunk) < max_seq_length:
                                chunk.append(eos_token_id)
                            else:
                                chunk[-1] = eos_token_id

                    # Pad if necessary
                    if len(chunk) < max_seq_length:
                        pad_value = pad_token if k == "input_ids" else 0
                        chunk = chunk + [pad_value] * (max_seq_length - len(chunk))

                    chunks.append(chunk)

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
        print(
            f"Process {current_ordinal}: Waiting {stagger_delay}s to stagger dataset loading..."
        )
        sleep(stagger_delay)

    # Load and tokenize dataset
    if args.pre_tokenized:
        print(
            f"Process {current_ordinal}: Loading pre-tokenized data - EOS tokens should already be present",
            flush=True,
        )
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
                f"Process {current_ordinal}: Loading pre-tokenized streaming dataset...",
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
                f"Process {current_ordinal}: Loading pre-tokenized dataset...",
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
            f"Process {current_ordinal}: Loading raw dataset for tokenization...",
            flush=True,
        )
        print(f"Dataset location: {args.dataset_dir}", flush=True)

        if args.training_file and args.validation_file:
            datasets = {"train": args.training_file, "validation": args.validation_file}
        else:
            train_loc = "train" if not args.debug else "validation"
            datasets = {
                "train": args.dataset_dir + f"/{train_loc}/*.{args.dataset_format}",
                "validation": args.dataset_dir + f"/validation/*.{args.dataset_format}",
            }

        if args.streaming_data:
            # Estimate steps/epoch only when loading from a directory
            if (not args.training_file) and (not args.validation_file):
                try:
                    if current_ordinal == 0:  # Only master process calculates this
                        train_loc = "train" if not args.debug else "validation"
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

        # For streaming data, don't use multiprocessing (num_proc=1 or omit)
        # Each TPU core is already a separate process
        print(f"Process {current_ordinal}: Tokenizing dataset...", flush=True)
        tokenize_fn = partial(
            tokenize_function,
            tokenizer=args.tokenizer,
            max_seq_length=args.max_seq_length,
            skip_eos=args.pre_tokenized,
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
            # Prefer the train split if available
            colnames = dataset["train"].column_names
        except Exception:
            try:
                # Fallback to dataset-level column names if present
                colnames = dataset.column_names
            except Exception:
                colnames = []
        safe_remove = [c for c in candidate_remove if c in (colnames or [])]

        # Remove num_proc for streaming to avoid conflicts
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

        print(
            f"Process {current_ordinal}: Performing chunking tokenized data...",
            flush=True,
        )
        group_fn = partial(
            group_texts,
            max_seq_length=args.max_seq_length,
            pad_token=args.tokenizer.pad_token_id,
            eos_token_id=args.tokenizer.eos_token_id,
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

    # Validate EOS token placement (only for master process to avoid duplicate output)
    if current_ordinal == 0 and not args.streaming_data:
        try:
            validate_eos_tokens(tokenized_dataset, args.tokenizer, num_samples=5)
        except Exception as e:
            print(f"EOS validation failed: {e}", flush=True)

    # Optional: Add a barrier to ensure all processes finish around the same time
    _barrier("dataset_prep_complete", xm.xla_device())

    return tokenized_dataset


def validate_eos_tokens(dataset, tokenizer, num_samples=10):
    """
    Validate that EOS tokens are properly placed in the dataset samples.

    Args:
        dataset: The tokenized dataset to validate
        tokenizer: The tokenizer used for tokenization
        num_samples: Number of samples to check (default: 10)

    Returns:
        dict: Validation results with statistics
    """
    print(f"Validating EOS token placement in {num_samples} samples...", flush=True)

    results = {
        "total_checked": 0,
        "samples_with_eos": 0,
        "samples_without_eos": 0,
        "eos_at_end_of_content": 0,
        "eos_in_padding": 0,
        "issues": [],
    }

    sample_count = 0

    # Handle both streaming and non-streaming datasets
    try:
        if hasattr(dataset, "take"):
            # Streaming dataset
            dataset_iter = dataset["train"].take(num_samples)
        else:
            # Regular dataset - take first num_samples
            dataset_iter = dataset["train"].select(
                range(min(num_samples, len(dataset["train"])))
            )
    except:
        # Fallback - try to iterate directly
        dataset_iter = dataset["train"]

    for batch in dataset_iter:
        if sample_count >= num_samples:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", [1] * len(input_ids))

        results["total_checked"] += 1
        sample_count += 1

        # Find where actual content ends (last non-pad token)
        content_end = len(input_ids) - 1
        while content_end >= 0 and input_ids[content_end] == tokenizer.pad_token_id:
            content_end -= 1

        # Check if EOS token is present
        has_eos = tokenizer.eos_token_id in input_ids
        if has_eos:
            results["samples_with_eos"] += 1
            eos_positions = [
                i
                for i, token in enumerate(input_ids)
                if token == tokenizer.eos_token_id
            ]

            # Check if EOS is at the end of content
            if content_end >= 0 and input_ids[content_end] == tokenizer.eos_token_id:
                results["eos_at_end_of_content"] += 1
            else:
                # EOS is in padding or middle of content
                if any(pos > content_end for pos in eos_positions):
                    results["eos_in_padding"] += 1
                    results["issues"].append(
                        f"Sample {sample_count}: EOS token found in padding area"
                    )
        else:
            results["samples_without_eos"] += 1
            results["issues"].append(f"Sample {sample_count}: No EOS token found")

    # Print validation summary
    print(f"EOS Token Validation Results:", flush=True)
    print(f"  Total samples checked: {results['total_checked']}", flush=True)
    print(f"  Samples with EOS token: {results['samples_with_eos']}", flush=True)
    print(f"  Samples without EOS token: {results['samples_without_eos']}", flush=True)
    print(f"  EOS at end of content: {results['eos_at_end_of_content']}", flush=True)
    print(f"  EOS in padding area: {results['eos_in_padding']}", flush=True)

    if results["issues"]:
        print(f"  Issues found: {len(results['issues'])}", flush=True)
        for issue in results["issues"][:5]:  # Show first 5 issues
            print(f"    {issue}", flush=True)

    return results


def test_eos_token_handling(tokenizer, max_seq_length=512):
    """
    Test function to demonstrate proper EOS token handling.

    Args:
        tokenizer: The tokenizer to test
        max_seq_length: Maximum sequence length for testing

    Returns:
        bool: True if tests pass, False otherwise
    """
    print("Testing EOS token handling...", flush=True)

    # Test cases
    test_texts = [
        "This is a short text.",
        "This is a longer text that should demonstrate how EOS tokens are properly added to sequences of various lengths.",
        "A" * (max_seq_length - 10),  # Near max length
        "B" * max_seq_length,  # At max length
        "C" * (max_seq_length + 50),  # Over max length
    ]

    all_tests_passed = True

    for i, text in enumerate(test_texts):
        print(f"Testing case {i + 1}: text length {len(text)}", flush=True)

        # Test tokenization
        examples = {"text": [text]}
        result = tokenize_function(examples, tokenizer, max_seq_length, skip_eos=False)

        input_ids = result["input_ids"][0]
        attention_mask = result["attention_mask"][0]

        # Find end of actual content (before padding)
        content_end = len(input_ids) - 1
        while content_end >= 0 and input_ids[content_end] == tokenizer.pad_token_id:
            content_end -= 1

        # Check if EOS token is at the end of content
        has_eos_at_end = (
            content_end >= 0 and input_ids[content_end] == tokenizer.eos_token_id
        )

        if has_eos_at_end:
            print(
                f"  ✓ EOS token correctly placed at end of content (position {content_end})",
                flush=True,
            )
        else:
            print(f"  ✗ EOS token missing or misplaced at end of content", flush=True)
            all_tests_passed = False

        # Check sequence length
        if len(input_ids) == max_seq_length:
            print(f"  ✓ Sequence properly padded to max_seq_length", flush=True)
        else:
            print(
                f"  ✗ Sequence length {len(input_ids)} != max_seq_length {max_seq_length}",
                flush=True,
            )
            all_tests_passed = False

    return all_tests_passed


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


def _barrier(tag, device):
    """Cross-host barrier using mesh_reduce (true ICI all-reduce).

    xm.rendezvous() may only synchronise processes on the *same* host on
    multi-host TPU pods.  mesh_reduce performs a real all-reduce over the
    ICI interconnect, so every rank across every host must participate
    before any of them can proceed.
    """
    dummy = torch.tensor([0.0], device=device)
    xm.mesh_reduce(tag, dummy, lambda x: sum(x))


# ── Background GCS uploader ─────────────────────────────────────────
# Keep a reference so we can join() before the next save to avoid
# racing two uploads for the same target path.
_gcs_upload_thread = None


def _upload_to_gcs_background(local_dir, gcs_output_dir, new_gcs_name):
    """Upload *local_dir* to GCS and rename in a background thread.

    The local directory is a persistent path (not a tempdir) so it
    survives after the calling function returns.  It is cleaned up
    here only after a successful upload.
    """
    upload_ok = False
    try:
        print(f"[GCS-upload] Uploading {local_dir} → {gcs_output_dir} …", flush=True)
        cp_result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", local_dir, gcs_output_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        if cp_result.stdout:
            print(f"[GCS-upload] cp stdout: {cp_result.stdout}", flush=True)
        if cp_result.stderr:
            print(f"[GCS-upload] cp stderr: {cp_result.stderr}", flush=True)

        uploaded_gcs_dir = f"{gcs_output_dir}/{os.path.basename(local_dir)}"
        new_gcs_path = f"{gcs_output_dir}/{new_gcs_name}"
        print(f"[GCS-upload] Renaming {uploaded_gcs_dir} → {new_gcs_path}", flush=True)
        mv_result = subprocess.run(
            ["gsutil", "-m", "mv", uploaded_gcs_dir, new_gcs_path],
            check=True,
            capture_output=True,
            text=True,
        )
        if mv_result.stdout:
            print(f"[GCS-upload] mv stdout: {mv_result.stdout}", flush=True)
        if mv_result.stderr:
            print(f"[GCS-upload] mv stderr: {mv_result.stderr}", flush=True)

        upload_ok = True
        print(f"[GCS-upload] Done: {new_gcs_path}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[GCS-upload] ERROR (gsutil returned {e.returncode}): {e}", flush=True)
        if e.stdout:
            print(f"[GCS-upload]   stdout: {e.stdout}", flush=True)
        if e.stderr:
            print(f"[GCS-upload]   stderr: {e.stderr}", flush=True)
        traceback.print_exc()
    except Exception as e:
        print(f"[GCS-upload] ERROR: {e}", flush=True)
        traceback.print_exc()
    finally:
        # Only clean up the local staging directory on success;
        # keep it around on failure so the user can retry manually.
        if upload_ok:
            try:
                shutil.rmtree(local_dir, ignore_errors=True)
            except Exception:
                pass
        else:
            print(
                f"[GCS-upload] WARNING: Upload failed. Local copy preserved at {local_dir}",
                flush=True,
            )


def _save_xla_model_to_dir(model, save_dir):
    """Save an XLA model to *save_dir* without copy.deepcopy().

    Instead of deep-copying the entire model graph and moving it to CPU
    (which triggers a huge XLA compilation + can OOM), we:
      1. Extract the state_dict (still XLA tensors – cheap, no copy).
      2. Convert each tensor to CPU one-by-one (small peak memory).
      3. Write config + safetensors via standard HF / safetensors APIs.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. state_dict → CPU tensors ──────────────────────────────────
    xm.mark_step()  # flush pending XLA ops
    sd = model.state_dict()  # lazy XLA tensors
    cpu_sd = {}
    for k, v in sd.items():
        cpu_sd[k] = v.cpu()  # materialise one tensor at a time
    del sd
    gc.collect()

    # ── 2. Write safetensors ─────────────────────────────────────────
    safetensors_path = os.path.join(save_dir, "model.safetensors")
    save_safetensors(cpu_sd, safetensors_path)
    del cpu_sd
    gc.collect()

    # ── 3. Write config.json (needed by from_pretrained) ─────────────
    model.config.save_pretrained(save_dir)
    print(f"[save] Model saved to {save_dir}", flush=True)


def train_fn(tokenized_dataset, device, args):
    # Initialize wandb for the master process
    print(f"Process {global_ordinal()}: Starting train_fn...", flush=True)
    if global_ordinal() == 0:  # .is_master_ordinal():
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
                "vocab_size": len(args.tokenizer),
                "pad_token": args.tokenizer.pad_token,
                "eos_token": args.tokenizer.eos_token,
                "pad_token_id": args.tokenizer.pad_token_id,
                "eos_token_id": args.tokenizer.eos_token_id,
            },
            mode="online",
            dir="/home/bes3/temp",
        )
        wandb.run.log_code(root="/home/bes3/models", name="cpt_llama")

    # Load model configuration
    # config = RobertaConfig.from_pretrained(args.model_name)

    # Load pre-trained model
    print(f"Process {global_ordinal()}: Loading the LM...", flush=True)
    if isinstance(args.checkpoint_path, str) & (args.checkpoint_path != ""):
        print(
            f"Process {global_ordinal()}: Loading model from checkpoint: {args.checkpoint_path}",
            flush=True,
        )

        # model_config = LlamaConfig.from_pretrained(args.model_name)
        config = LlamaConfig.from_pretrained(
            args.model_name, token=args.huggingface_token
        )

        # Create model with config (avoids meta parameters)
        with torch.device("cpu"):
            model = LlamaForCausalLM(config)
        print(f"Process {global_ordinal()}: Empty model created on CPU", flush=True)

        if args.checkpoint_path.startswith("gs://"):
            # Parse GCS path
            print(f"Loading model from: {args.checkpoint_path}...", flush=True)
            bucket_name = args.checkpoint_path.split("/")[2]
            blob_name = "/".join(args.checkpoint_path.split("/")[3:])
            local_path = f"/tmp/checkpoint.{'/'.join(args.checkpoint_path.split('/')[-2:])}"  # Temporary local path to store the downloaded file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading to {local_path}", flush=True)
            checkpoint = load_from_gcs(bucket_name, blob_name, local_path, device)
            sleep(1)
            print(f"Process {global_ordinal()}: Checkpoint downloaded...", flush=True)
            _barrier("checkpoint_downloaded", device)
        else:
            if args.checkpoint_path.endswith(".safetensors"):
                checkpoint = load_safetensors(args.checkpoint_path, device=device)
            else:
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
            sleep(1)
            print(f"Process {global_ordinal()}: Checkpoint downloaded...", flush=True)
            _barrier("checkpoint_downloaded", device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        sleep(1)
        print(f"Process {global_ordinal()}: Checkpoint loaded in memory..", flush=True)
    else:
        with torch.device("cpu"):
            model = LlamaForCausalLM.from_pretrained(
                args.model_name,
                token=args.huggingface_token,
                device_map=None,
                _fast_init=False,
                low_cpu_mem_usage=False,
            )

    model.config.pad_token_id = args.tokenizer.pad_token_id

    _barrier("checkpoint_loaded_in_memory", device)

    # Debug single core mode
    print(
        f"Process {global_ordinal()}: world_size={world_size()}, global_ordinal={global_ordinal()}, num_cores={args.num_cores}",
        flush=True,
    )

    # For single core mode, simplify the model loading
    if args.num_cores == 1:
        print(
            f"Process {global_ordinal()}: Single core mode - loading model to TPU device {device}...",
            flush=True,
        )
        try:
            model = model.to(device=device, dtype=torch.bfloat16)
            xm.mark_step()  # Ensure model transfer completes
            print(
                f"Process {global_ordinal()}: Model loaded to device successfully",
                flush=True,
            )
            print(
                f"Process {global_ordinal()}: Model device after loading: {next(model.parameters()).device}",
                flush=True,
            )
        except Exception as e:
            print(
                f"Process {global_ordinal()}: Error loading model to device: {e}",
                flush=True,
            )
            raise
    else:
        # Original multi-core logic
        for i in range(world_size()):
            if global_ordinal() == i:
                print(
                    f"Process {global_ordinal()}: My turn to load model to TPU device {device}...",
                    flush=True,
                )
                try:
                    model = model.to(device=device, dtype=torch.bfloat16)
                    # model.gradient_checkpointing_enable()
                    print(
                        f"Process {global_ordinal()}: Checkpoint loaded on device successfully",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"Process {global_ordinal()}: Error loading model to device: {e}",
                        flush=True,
                    )
                    raise

            # Wait for current process to finish before next one starts
            _barrier(f"model_to_device_{i}", device)

    print(f"Process {global_ordinal()}: All models loaded to devices", flush=True)

    # print("After checkpointing, model on device:", next(model.parameters()).device, flush=True)
    # if hasattr(model, 'tie_weights'):
    #     model.tie_weights()
    # Set up data collator
    print(f"Process: {global_ordinal()}. Setting up data collator.")
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer, mlm=False)
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

        # For single process mode, use process-based indices instead of TPU core indices
        if args.num_cores == 1:
            num_shards = 1
            shard_id = 0
        else:
            num_shards = world_size()  # Total number of TPU cores (or processes)
            shard_id = global_ordinal()  # Unique id for the current process
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
            print(
                f"Process {global_ordinal()}: Fallback sharded dataset created successfully",
                flush=True,
            )

        print(
            f"Process {global_ordinal()}: Creating streaming DataLoader...", flush=True
        )

        train_dataloader = torch.utils.data.DataLoader(
            sharded_train_dataset,
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

    # For single core, try alternative data loading approach
    if args.num_cores == 1:
        print(f"Process {global_ordinal()}: Using single core data loading", flush=True)
        # For single core, we might not need MpDeviceLoader
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

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # optimizer = torch.optim.Adafactor(model.parameters(), lr=args.learning_rate, #weight_decay=args.weight_decay, scale_parameter=False, relative_step=False)

    steps_per_epoch = (
        args.max_steps_per_epoch if args.streaming_data else len(train_dataloader)
    )
    save_steps = (
        int(steps_per_epoch * args.save_epoch_percentage)
        if args.save_epoch_percentage < 1
        else int(args.save_epoch_percentage)
    )
    # Align save_steps to the nearest accumulation boundary so that
    # total_step (which increments by gradient_accumulation_steps) can
    # actually hit it.  Without this, total_step % save_steps may never
    # be zero if the two are not aligned.
    _gas = args.gradient_accumulation_steps
    if _gas > 1 and save_steps % _gas != 0:
        save_steps = max(_gas, (save_steps // _gas) * _gas)
    # Guard against save_steps == 0 (would cause modulo-by-zero)
    save_steps = max(save_steps, _gas)

    # Warn if logging_steps is not aligned with gradient_accumulation_steps.
    # Logging now happens INSIDE the accumulation-boundary block, so the
    # condition (step + 1) % logging_steps == 0 can only fire when
    # (step + 1) is also a multiple of gradient_accumulation_steps.
    # If logging_steps is not a multiple of GAC, logging will never trigger.
    if _gas > 1 and args.logging_steps % _gas != 0:
        aligned_logging = max(_gas, (args.logging_steps // _gas) * _gas)
        print(
            f"[WARNING] logging_steps={args.logging_steps} is not a multiple of "
            f"gradient_accumulation_steps={_gas}. Aligning to {aligned_logging} "
            f"so that logging actually fires at accumulation boundaries.",
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
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)

    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=args.num_cycles,
        )

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

    _barrier("start_training", device)
    if global_ordinal() == 0:  # xm.is_master_ordinal():
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
        # (optimizer_step, mesh_reduce, rendezvous) until ALL cores agree to
        # stop.  This prevents the cross-deadlock that occurs when one core
        # exits the loop while others block on a collective op.
        data_iter = iter(safe_iter(xla_train_loader))
        local_exhausted = False
        _diag = args.debug

        while True:
            step += 1

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
            #     ANY core is done.  This is the earliest safe place to
            #     break because optimizer_step (the next collective op)
            #     has not been entered yet. ---
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

            # --- exhausted core: still participate in every collective op
            #     that the training body would have executed this step.
            #     The collective section below is now shared with the
            #     normal-training path (it is OUTSIDE the try/except),
            #     so an exhausted core only needs to zero its grads and
            #     then jump straight to the shared collective block. ---
            if local_exhausted:
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} exhausted-path",
                        flush=True,
                    )
                optimizer.zero_grad()
                step_ok = False
                # Jump past the forward/backward block straight to
                # the collective section (which starts after this
                # if/else chain).
            else:
                # --- normal training step (batch is not None) ---
                assert batch is not None, "batch should never be None here"

                # ── forward / backward in a narrow try-block ──────────
                # Collective ops (optimizer_step, mesh_reduce, barriers) are
                # OUTSIDE this try so they are NEVER skipped.  Skipping a
                # collective on one rank while others enter it is the #1
                # cause of cross-host deadlocks.
                step_ok = True  # flipped to False if forward/bwd fails
                try:
                    # Debug tensor information for first few batches
                    if step < 3:
                        print(
                            f"Process {global_ordinal()}: Processing batch {step}...",
                            flush=True,
                        )

                    if step == 0:
                        print(
                            f"Process {global_ordinal()}: First batch tensor info:",
                            flush=True,
                        )
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                is_xla = "xla" in str(v.device).lower()
                                print(
                                    f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, is_xla={is_xla}",
                                    flush=True,
                                )

                        # Check model device
                        model_device = next(model.parameters()).device
                        print(f"  Model parameters device: {model_device}", flush=True)
                        print(f"  Target device: {device}", flush=True)

                        # Verify XLA is working
                        try:
                            test_tensor = torch.tensor([1.0]).to(device)
                            print(
                                f"  Test tensor device: {test_tensor.device}, is_xla: {'xla' in str(test_tensor.device).lower()}",
                                flush=True,
                            )
                        except Exception as tensor_e:
                            print(
                                f"  Error creating test tensor: {tensor_e}", flush=True
                            )

                        # Verify model embeddings are on XLA device
                        try:
                            embedding_device = model.model.embed_tokens.weight.device
                            print(
                                f"  Model embeddings device: {embedding_device}, is_xla: {'xla' in str(embedding_device).lower()}",
                                flush=True,
                            )

                            # For single core, ensure model is properly on XLA device
                            if (
                                args.num_cores == 1
                                and "xla" not in str(embedding_device).lower()
                            ):
                                print(
                                    f"  WARNING: Model not on XLA device in single core mode, forcing move...",
                                    flush=True,
                                )
                                model = model.to(device)
                                xm.mark_step()  # Ensure transfer completes
                                new_embedding_device = (
                                    model.model.embed_tokens.weight.device
                                )
                                print(
                                    f"  Model moved - new embeddings device: {new_embedding_device}",
                                    flush=True,
                                )
                        except AttributeError as ae:
                            print(
                                f"  Could not access model embeddings: {ae}", flush=True
                            )

                    # Handle tensor placement - simplified approach for single core
                    if args.num_cores == 1:
                        # For single core, be more explicit about tensor movement
                        new_batch = {}
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                # Ensure tensor is on XLA device
                                if not ("xla" in str(v.device).lower()):
                                    if v.dtype == torch.float32:
                                        new_batch[k] = v.to(
                                            device, dtype=torch.bfloat16
                                        )
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
                                    print(
                                        f"    {k}: {v.device} (shape: {v.shape})",
                                        flush=True,
                                    )

                            # Ensure XLA operations are synchronized before model forward pass
                            xm.mark_step()
                    else:
                        # Original logic for multi-core
                        batch = {
                            k: v.to(device=device, dtype=torch.bfloat16)
                            if v.dtype == torch.float32
                            else v.to(device)
                            for k, v in batch.items()
                        }

                    loss = model(**batch).loss / args.gradient_accumulation_steps

                    # Unconditionally replace NaN/Inf — no data-dependent branch.
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                    loss.backward()

                    # ── Accumulate loss as an XLA tensor ──────────────
                    # AVOID .item() on every micro-step!  Each .item() forces
                    # eager graph compilation + device→host transfer, which
                    # fragments the XLA compilation cache and slowly exhausts
                    # HBM over thousands of steps.  Instead we keep a running
                    # XLA-side sum and only materialise at logging boundaries.
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
                    # Clear any partially-accumulated gradients so the
                    # upcoming optimizer_step is a harmless no-op for this
                    # rank, but the collective still executes.
                    optimizer.zero_grad()

            # ══════════════════════════════════════════════════════
            # ── Collective operations — ALWAYS executed ───────────
            # Both the exhausted-path and the normal-training path
            # (whether the try succeeded or failed) converge here.
            # These are outside the try/except so that even if one
            # rank had an exception above, it still participates in
            # every collective.  Skipping any of these causes a
            # cross-host deadlock because the other ranks block
            # waiting for this rank to join the all-reduce.
            # ══════════════════════════════════════════════════════

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if _diag:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} ENTER grad_clip",
                        flush=True,
                    )
                # Single clip call (the second was redundant)
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
                # where all ranks across all hosts are provably
                # synchronized.  Placing them at arbitrary micro-steps
                # causes cross-host deadlocks because ranks on
                # different hosts run micro-steps independently and
                # may not reach the collective at the same time.
                # ══════════════════════════════════════════════════

                # --- Logging ---
                if (step + 1) % args.logging_steps == 0:
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} ENTER mesh_reduce(loss)",
                        flush=True,
                    )
                    # Materialise the XLA-side running sums only at
                    # logging boundaries (once every N steps, not every
                    # micro-step).  This is the ONE place we call .item().
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
                    # Replace the XLA tensors with plain floats so they
                    # don't pin old graph nodes in memory.
                    total_loss = total_loss_val
                    sub_total_loss = sub_total_loss_val

                    local_avg_loss = total_loss_val / (step + 1)
                    local_avg_loss_N = sub_total_loss_val / max(sub_step, 1)

                    global_avg_loss = xm.mesh_reduce(
                        f"loss_s{step}", local_avg_loss, np.mean
                    )
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} EXIT mesh_reduce(loss)",
                        flush=True,
                    )
                    global_avg_loss_N = xm.mesh_reduce(
                        f"loss_N_s{step}", local_avg_loss_N, np.mean
                    )
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} EXIT mesh_reduce(loss_N)",
                        flush=True,
                    )

                    perplexity = math.exp(global_avg_loss)
                    perplexity_N = math.exp(global_avg_loss_N)

                    sub_step = 0
                    sub_total_loss = 0.0

                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} ENTER barrier(before_wandb_log)",
                        flush=True,
                    )
                    _barrier(f"before_wandb_log_s{step}", device)
                    print(
                        f"[DIAG][Rank {global_ordinal()}] step={step} EXIT barrier(before_wandb_log)",
                        flush=True,
                    )
                    if global_ordinal() == 0:
                        print(
                            f"Logging for epoch: {epoch}, step: {step}, train_perplexity_N:{perplexity_N}",
                            flush=True,
                        )
                        wandb.log(
                            {
                                "train_global_average_loss": global_avg_loss,
                                "train_global_average_loss_N": global_avg_loss_N,
                                "train_perplexity": perplexity,
                                "train_perplexity_N": perplexity_N,
                                "epoch": epoch,
                                "step": step,
                                "total_step": total_step,
                            },
                            commit=True,
                        )
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} ENTER barrier(after_wandb_log)",
                            flush=True,
                        )
                    _barrier(f"after_wandb_log_s{step}", device)
                    if _diag:
                        print(
                            f"[DIAG][Rank {global_ordinal()}] step={step} EXIT barrier(after_wandb_log)",
                            flush=True,
                        )

                # --- Mid-epoch checkpoint saving ---
                # Use total_step (in optimizer-step units) so the
                # condition is step-based and fires only at
                # accumulation boundaries where ranks are synced.
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

                    # ── XLA-safe model saving ────────────────────────
                    # All ranks participate up to the barrier; only
                    # rank 0 does the actual I/O.  The GCS upload is
                    # kicked off in a background thread so that the
                    # after-save barrier is reached quickly and other
                    # ranks are not stuck waiting for a multi-GB upload.
                    global _gcs_upload_thread

                    if global_ordinal() == 0:
                        print(f"Saving model at total_step={total_step}...", flush=True)
                        wandb.log({"epoch": epoch, "step": total_step}, commit=True)

                        # Wait for any previous background upload to finish
                        # before we start writing to a new staging dir.
                        if (
                            _gcs_upload_thread is not None
                            and _gcs_upload_thread.is_alive()
                        ):
                            print(
                                "[save] Waiting for previous GCS upload to finish…",
                                flush=True,
                            )
                            _gcs_upload_thread.join()

                        if args.output_dir.startswith("gs://"):
                            # Use a persistent staging dir (not a tempdir)
                            # so it survives into the background thread.
                            staging_dir = os.path.join(
                                args.tmp_dir, f"ckpt_step_{total_step}"
                            )
                            _save_xla_model_to_dir(model, staging_dir)

                            # Kick off the GCS upload in a background thread
                            new_dir_name = f"{args.model_name}_latest"
                            _gcs_upload_thread = threading.Thread(
                                target=_upload_to_gcs_background,
                                args=(staging_dir, args.output_dir, new_dir_name),
                                daemon=False,
                            )
                            _gcs_upload_thread.start()
                        else:
                            _save_xla_model_to_dir(model, args.output_dir)

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

        # All cores arrive here together (either via the epoch_stop_check
        # break or via the steps_per_epoch break).
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

            # Wait for any previous background upload before starting a new one
            if _gcs_upload_thread is not None and _gcs_upload_thread.is_alive():
                print("[save] Waiting for previous GCS upload to finish…", flush=True)
                _gcs_upload_thread.join()

            if args.output_dir.startswith("gs://"):
                staging_dir = os.path.join(args.tmp_dir, f"ckpt_epoch_{epoch}")
                _save_xla_model_to_dir(model, staging_dir)

                # Epoch-end saves are important — upload synchronously
                # so we are sure it completes before the next epoch.
                current_time = datetime.datetime.now().strftime("%Y%m%d")
                new_dir_name = f"{args.model_name}_epoch{epoch}_{current_time}"
                _upload_to_gcs_background(staging_dir, args.output_dir, new_dir_name)
            else:
                _save_xla_model_to_dir(model, args.output_dir)

        # Epoch-end save barrier — all ranks must wait for rank 0's save
        _barrier(f"end_epoch_{epoch}_after_save", device)

    # Make sure any pending GCS upload finishes before the process exits
    if global_ordinal() == 0:
        if _gcs_upload_thread is not None and _gcs_upload_thread.is_alive():
            print(
                "[save] Waiting for final GCS upload to finish before exit…", flush=True
            )
            _gcs_upload_thread.join()
            print("[save] Final GCS upload thread finished.", flush=True)

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

    # Rank 0 runs EOS token handling test after XLA runtime is initialized
    if global_ordinal() == 0:
        try:
            test_passed = test_eos_token_handling(args.tokenizer, args.max_seq_length)
            print(
                f"EOS token handling test: {'PASSED' if test_passed else 'FAILED'}",
                flush=True,
            )
        except Exception as e:
            print(f"EOS token test failed with error: {e}", flush=True)

    _barrier("prep_start", device)
    tokenized_dataset = prep_fn(args)

    _barrier("train_start", device)
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
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
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
    parser.add_argument(
        "--training_file",
        type=str,
        default=None,
        help="Path to training parquet file (local path or gs://)",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Path to validation parquet file (local path or gs://)",
    )
    parser.add_argument(
        "--dataset_format", default="json", choices=["json", "parquet", "arrow"]
    )
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pre_tokenized", action="store_true", default=False)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["linear", "cosine"],
        default="linear",
        help="linear=warmup→decay; oscillatory=cosine‐restarts",
    )
    parser.add_argument(
        "--num_cycles",
        type=float,
        default=10,
        help="number of cycles for oscillatory schedule (defaults to 10)",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="minimum LR for oscillatory schedule",
    )
    parser.add_argument("--max_grad_norm", type=float, default=14.4)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--logging_steps", type=int, default=1_000)
    parser.add_argument("--save_epoch_percentage", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cores", type=int, default=4)
    parser.add_argument("--keep_in_memory", action="store_true")
    parser.add_argument("--streaming_data", action="store_true")
    parser.add_argument("--sharded_data", action="store_true")
    parser.add_argument("--max_steps_per_epoch", type=int, default=50_000_000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=1_000)
    parser.add_argument(
        "--shuffle_dataset_path", type=str, default="/home/bob/tmp/shuffle.parquet"
    )
    parser.add_argument("--shuffle_dataset_ext", type=str, default=None)
    parser.add_argument("--shuffle_dataset", action="store_true")
    parser.add_argument("--shuffle_force_update", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument(
        "--model_name", type=str, default="UMCU/CardioLlama.nl_clinical"
    )
    parser.add_argument(
        "--wandb_key", type=str, required=True, help="Weights & Biases API key"
    )
    parser.add_argument("--huggingface_token", type=str, required=True)
    parser.add_argument(
        "--TPU_NAME", type=str, default="unknown, please set with --TPU_ID"
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

    # Set the same seed for all processes
    torch.manual_seed(args.seed)

    # Using explicit --training_file/--validation_file with streaming; shuffle-related logic removed.

    args.tokenizer = LlamaTokenizerFast.from_pretrained(
        args.tokenizer_name_or_path, token=args.huggingface_token
    )
    args.tokenizer.model_max_length = args.max_seq_length
    if args.tokenizer.pad_token is None:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    # Ensure EOS token is properly set
    if args.tokenizer.eos_token is None:
        print("Warning: No EOS token found in tokenizer, using default", flush=True)
        args.tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
    if args.tokenizer.bos_token is None:
        print("Warning: No BOS token found in tokenizer, using default", flush=True)
        args.tokenizer.add_special_tokens({"bos_token": "<|begin_of_text|>"})

    print(
        f"Tokenizer EOS token: {args.tokenizer.eos_token} (ID: {args.tokenizer.eos_token_id})",
        flush=True,
    )

    # EOS test moved into prep_and_train_fn(rank 0) to avoid pre-spawn XLA init

    # Using training/validation files; no shuffle dataset redirection.

    sleep(5)

    # Spawn processes
    print(f"Initializing TPU...with {args.num_cores} cores")
    print("Spawning processes...")

    if args.num_cores == 1:
        print("Running single process...", flush=True)
        prep_and_train_fn(0, args)
    else:
        # set TPU_NUM_DEVICES to args.num_cores
        os.environ["TPU_NUM_DEVICES"] = str(args.num_cores)
        # Ensure XLA uses all available cores
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
