"""TPU training script for a NeoBERT-style masked language model with GCS parquet support.

This script is adapted for loading datasets from GCS parquet files instead of
HuggingFace datasets. It uses PyTorch XLA for TPU training and supports streaming
from multiple parquet files stored in Google Cloud Storage.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional
import glob

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


@dataclass
class TrainingArgs:
    model_dir: str
    tokenizer_path: str
    config_path: str
    dataset_path: str
    validation_dataset_path: Optional[str]
    text_column: str
    batch_size: int
    vocab_size: int
    max_length: int
    mlm_probability: float
    pad_to_multiple: int
    pack_sequences: bool
    mask_all: bool
    learning_rate: float
    weight_decay: float
    num_epochs: int
    log_every: int
    warmup_ratio: float
    grad_clip: float
    seed: int
    neobert_src: str | None
    streaming: bool
    shuffle_buffer_size: int
    steps_per_epoch: Optional[int]
    pretrained_model_path: Optional[str]
    local_dataset_cache: Optional[str]
    validation_cache_path: Optional[str]
    validation_steps: Optional[int]
    eval_every_n_epochs: int


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description="TPU NeoBERT training script with GCS parquet support")
    parser.add_argument("--model_dir", type=str, required=True, help="Output directory for checkpoints and logs.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path or repo ID for the custom tokenizer.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path or repo ID for the base NeoBERT config JSON (vocab size will be overridden).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="GCS path to directory containing parquet files (e.g., 'gs://bucket/dataset/train').",
    )
    parser.add_argument(
        "--validation_dataset_path",
        type=str,
        default=None,
        help="GCS path to validation parquet file or directory (e.g., 'gs://bucket/validation').",
    )
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing raw text.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-core batch size.")
    parser.add_argument("--vocab_size", type=int, default=100_000, help="Vocabulary size to set on config/tokenizer.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization/packing.")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--pad_to_multiple", type=int, default=8, help="Pad sequences to a multiple of this value.")
    parser.add_argument("--pack_sequences", action="store_true", help="Enable NeoBERT sequence packing collator.")
    parser.add_argument("--mask_all", action="store_true", help="Use 100% masking instead of 80/10/10.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--decay_rate", type=float, default=0.01, help="Weight decay rate.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--log_every", type=int, default=50, help="Logging frequency (in steps).")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Portion of total steps used for LR warmup (0-1).",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--neobert_src",
        type=str,
        default=None,
        help="Optional path to the NeoBERT repo (folder containing src/) when not installed as a package.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming dataset loading with buffered shuffling to avoid in-memory shuffles.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="Shuffle buffer size used when streaming datasets.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Required when using streaming datasets without a known length to size the scheduler/epoch loop.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Optional path or HF repo id to load a pretrained NeoBERT checkpoint for continued pretraining.",
    )
    parser.add_argument(
        "--local_dataset_cache",
        type=str,
        default="/tmp/dataset_cache",
        help="Local directory to cache downloaded parquet files.",
    )
    parser.add_argument(
        "--validation_cache_path",
        type=str,
        default="/tmp/validation_cache",
        help="Local directory to cache validation parquet files.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Number of validation steps to run per evaluation.",
    )
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )

    args = parser.parse_args()
    return TrainingArgs(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        dataset_path=args.dataset_path,
        validation_dataset_path=args.validation_dataset_path,
        text_column=args.text_column,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        pad_to_multiple=args.pad_to_multiple,
        pack_sequences=args.pack_sequences,
        mask_all=args.mask_all,
        learning_rate=args.learning_rate,
        weight_decay=args.decay_rate,
        num_epochs=args.num_epochs,
        log_every=args.log_every,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        seed=args.seed,
        neobert_src=args.neobert_src,
        streaming=args.streaming,
        shuffle_buffer_size=args.shuffle_buffer_size,
        steps_per_epoch=args.steps_per_epoch,
        pretrained_model_path=args.pretrained_model_path,
        local_dataset_cache=args.local_dataset_cache,
        validation_cache_path=args.validation_cache_path,
        validation_steps=args.validation_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def load_tokenizer(path: str, args: TrainingArgs) -> AutoTokenizer:
    from neobert.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(pretrained_model_name_or_path=path, vocab_size=args.vocab_size, max_length=args.max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def resolve_neobert_import_path(path: str | None) -> None:
    if path is None:
        return
    candidate = os.path.join(path, "src") if not path.endswith("src") else path
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def download_gcs_parquet_files(gcs_path: str, local_cache_dir: str) -> List[str]:
    """Download parquet files from GCS to local cache directory."""
    import subprocess

    # Create local cache directory
    os.makedirs(local_cache_dir, exist_ok=True)

    # Use gsutil to list and copy parquet files
    try:
        # List all parquet files in the GCS path
        result = subprocess.run(
            ["gsutil", "ls", f"{gcs_path}/*.parquet"],
            capture_output=True,
            text=True,
            check=True
        )
        gcs_files = result.stdout.strip().split('\n')
        gcs_files = [f for f in gcs_files if f.endswith('.parquet')]

        local_files = []
        for gcs_file in gcs_files:
            # Extract filename from GCS path
            filename = os.path.basename(gcs_file)
            local_file = os.path.join(local_cache_dir, filename)

            # Download if not already cached
            if not os.path.exists(local_file):
                xm.master_print(f"Downloading {gcs_file} to {local_file}")
                subprocess.run(
                    ["gsutil", "cp", gcs_file, local_file],
                    check=True
                )
            else:
                xm.master_print(f"Using cached file: {local_file}")

            local_files.append(local_file)

        return sorted(local_files)

    except subprocess.CalledProcessError as e:
        xm.master_print(f"Error downloading from GCS: {e}")
        raise


def load_parquet_dataset_streaming(parquet_files: List[str], text_column: str, seed: int, buffer_size: int):
    """Load parquet files as streaming dataset."""
    # For streaming, we'll create a custom iterable dataset
    def data_generator():
        import random
        random.seed(seed)

        # Shuffle the file order
        shuffled_files = parquet_files.copy()
        random.shuffle(shuffled_files)

        for parquet_file in shuffled_files:
            df = pd.read_parquet(parquet_file)
            if text_column not in df.columns:
                xm.master_print(f"Warning: Column '{text_column}' not found in {parquet_file}")
                continue

            # Convert to list and shuffle
            texts = df[text_column].tolist()
            random.shuffle(texts)

            for text in texts:
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    yield {text_column: text}

    # Create streaming dataset
    from datasets import IterableDataset
    dataset = IterableDataset.from_generator(data_generator)

    # Apply shuffling if requested
    if buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    return dataset


def load_parquet_dataset_in_memory(parquet_files: List[str], text_column: str):
    """Load all parquet files into memory as a single dataset."""
    all_data = []

    for parquet_file in parquet_files:
        xm.master_print(f"Loading {parquet_file}")
        df = pd.read_parquet(parquet_file)

        if text_column not in df.columns:
            xm.master_print(f"Warning: Column '{text_column}' not found in {parquet_file}")
            continue

        # Filter out empty/null texts
        valid_texts = df[text_column].dropna()
        valid_texts = valid_texts[valid_texts.str.strip() != '']

        all_data.extend([{text_column: text} for text in valid_texts.tolist()])

    xm.master_print(f"Loaded {len(all_data)} text samples from {len(parquet_files)} files")
    return Dataset.from_list(all_data)


def load_validation_dataset(validation_path: str, cache_path: str, text_column: str):
    """Load validation dataset from GCS parquet file(s)."""
    if validation_path is None:
        return None

    # Download validation file(s)
    if xm.is_master_ordinal():
        val_files = download_gcs_parquet_files(validation_path, cache_path)
    else:
        import time
        time.sleep(5)
        val_files = sorted(glob.glob(os.path.join(cache_path, "*.parquet")))

    if not val_files:
        xm.master_print("No validation files found")
        return None

    # Load validation dataset in memory (usually smaller than training)
    return load_parquet_dataset_in_memory(val_files, text_column)


def evaluate_model(model, val_dataloader, loss_fn, vocab_size, max_steps=None):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if max_steps and step >= max_steps:
                break

            logits = model(batch["input_ids"], batch.get("attention_mask", None))["logits"]
            loss = loss_fn(logits.view(-1, vocab_size), batch["labels"].view(-1))
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    model.train()
    return avg_loss


def build_dataloader(dataset, tokenizer: AutoTokenizer, args: TrainingArgs) -> DataLoader:
    # Lazy import to avoid requiring NeoBERT when not used.
    from neobert.collator import get_collator

    data_collator = get_collator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=args.pad_to_multiple,
        mask_all=args.mask_all,
        pack_sequences=args.pack_sequences,
        max_length=args.max_length,
    )

    is_streaming = hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')
    if is_streaming:
        # For streaming datasets, shard across TPU cores
        dataset = dataset.shard(num_shards=xm.xrt_world_size(), index=xm.get_ordinal())
        sampler = None
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
            drop_last=True,
        )

    def tokenize_examples(batch):
        return tokenizer(batch[args.text_column], truncation=True, padding=False, max_length=args.max_length)

    dataset = dataset.map(tokenize_examples, batched=True, remove_columns=[args.text_column])
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=True,
    )


def build_model(tokenizer, args: TrainingArgs):
    from neobert.model import NeoBERTConfig, NeoBERTLMHead

    if args.pretrained_model_path:
        model = NeoBERTLMHead.from_pretrained(args.pretrained_model_path)
        # Resize if a different vocab size was requested to align with tokenizer.
        if model.config.vocab_size != args.vocab_size:
            model.resize_token_embeddings(args.vocab_size)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.max_length = args.max_length
    else:
        config = NeoBERTConfig.from_pretrained(args.config_path)
        config.vocab_size = args.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.max_length = args.max_length
        model = NeoBERTLMHead(config)
    return model


def create_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int, args: TrainingArgs):
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def train_worker(index: int, args: TrainingArgs):
    set_seed(args.seed + index)

    resolve_neobert_import_path(args.neobert_src)
    tokenizer = load_tokenizer(args.tokenizer_path, args)

    # Download parquet files from GCS
    if xm.is_master_ordinal():
        parquet_files = download_gcs_parquet_files(args.dataset_path, args.local_dataset_cache)
    else:
        # Non-master processes wait and then scan the cache directory
        import time
        time.sleep(10)  # Give master time to download
        parquet_files = sorted(glob.glob(os.path.join(args.local_dataset_cache, "*.parquet")))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {args.dataset_path}")

    xm.master_print(f"Found {len(parquet_files)} training parquet files")

    # Load training dataset
    if args.streaming:
        dataset = load_parquet_dataset_streaming(
            parquet_files,
            args.text_column,
            args.seed + index,
            args.shuffle_buffer_size
        )
    else:
        dataset = load_parquet_dataset_in_memory(parquet_files, args.text_column)

    dataloader = build_dataloader(dataset, tokenizer, args)

    # Load validation dataset if provided
    val_dataloader = None
    if args.validation_dataset_path:
        val_dataset = load_validation_dataset(
            args.validation_dataset_path,
            args.validation_cache_path,
            args.text_column
        )
        if val_dataset:
            val_dataloader = build_dataloader(val_dataset, tokenizer, args)
            xm.master_print(f"Loaded validation dataset with {len(val_dataset)} samples")

    device = xm.xla_device()
    model = build_model(tokenizer, args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        if args.steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be set when using streaming datasets without a known length.")
        steps_per_epoch = args.steps_per_epoch

    num_training_steps = math.ceil(steps_per_epoch * args.num_epochs)
    scheduler = create_scheduler(optimizer, num_training_steps, args)

    xm.master_print(f"Starting training on {xm.xrt_world_size()} TPU cores")
    xm.master_print(f"Steps per epoch: {steps_per_epoch}")
    xm.master_print(f"Total training steps: {num_training_steps}")

    model.train()

    for epoch in range(args.num_epochs):
        xm.master_print(f"Epoch {epoch + 1}/{args.num_epochs}")
        if dataloader.sampler is not None:
            dataloader.sampler.set_epoch(epoch)
        if hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch)
        tracker = xm.RateTracker()
        device_loader = pl.MpDeviceLoader(dataloader, device)

        for step, batch in enumerate(device_loader, start=1):
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch.get("attention_mask", None))["logits"]
            loss = loss_fn(logits.view(-1, args.vocab_size), batch["labels"].view(-1))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(optimizer)
            scheduler.step()

            tracker.add(args.batch_size)
            if step % args.log_every == 0:
                xm.master_print(
                    f"Epoch {epoch + 1} Step {step}: loss={loss.item():.4f} | rate={tracker.rate():.2f} samples/sec"
                )
            if step >= steps_per_epoch:
                break

        # Run validation if available and appropriate
        if (val_dataloader is not None and
            (epoch + 1) % args.eval_every_n_epochs == 0):
            xm.master_print(f"Running validation after epoch {epoch + 1}")
            val_device_loader = pl.MpDeviceLoader(val_dataloader, device)
            val_loss = evaluate_model(model, val_device_loader, loss_fn, args.vocab_size, args.validation_steps)
            xm.master_print(f"Validation loss: {val_loss:.4f}")

        xm.master_print(f"Finished epoch {epoch + 1}")

    if xm.is_master_ordinal():
        os.makedirs(args.model_dir, exist_ok=True)
        tokenizer.save_pretrained(args.model_dir)
        xm.save(model.state_dict(), os.path.join(args.model_dir, "pytorch_model.bin"))
        model.config.to_json_file(os.path.join(args.model_dir, "config.json"))
        xm.master_print(f"Model and tokenizer saved to {args.model_dir}")


def main():
    args = parse_args()
    xmp.spawn(train_worker, args=(args,), nprocs=8, start_method="fork")


if __name__ == "__main__":
    main()
