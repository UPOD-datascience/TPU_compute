"""Simple TPU training script for a NeoBERT-style masked language model.

This script assumes a custom tokenizer and a config similar to the one in
https://huggingface.co/chandar-lab/NeoBERT/blob/main/config.json with a
vocabulary size overridden to 100,000. It uses PyTorch XLA for TPU training,
and supports multiple Hugging Face datasets provided as a comma-separated list.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


@dataclass
class TrainingArgs:
    model_dir: str
    tokenizer_path: str
    config_path: str
    dataset_names: List[str]
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
    dataset_probabilities: Optional[List[float]]
    streaming: bool
    shuffle_buffer_size: int
    steps_per_epoch: Optional[int]
    pretrained_model_path: Optional[str]


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description="TPU NeoBERT training script")
    parser.add_argument("--model_dir", type=str, required=True, help="Output directory for checkpoints and logs.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path or repo ID for the custom tokenizer.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path or repo ID for the base NeoBERT config JSON (vocab size will be overridden).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="X",
        help="Comma-separated list of Hugging Face dataset identifiers (e.g., 'X,Y,Z').",
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
        "--dataset_probs",
        type=str,
        default=None,
        help="Comma-separated probabilities (aligned with --dataset_name) to interleave datasets instead of concatenating.",
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

    args = parser.parse_args()
    dataset_names = [name.strip() for name in args.dataset_name.split(",") if name.strip()]
    dataset_probabilities = None
    if args.dataset_probs:
        dataset_probabilities = [float(p.strip()) for p in args.dataset_probs.split(",") if p.strip()]
        if len(dataset_probabilities) != len(dataset_names):
            parser.error("Number of values in --dataset_probs must match the number of datasets in --dataset_name.")
    return TrainingArgs(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        dataset_names=dataset_names,
        dataset_probabilities=dataset_probabilities,
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


def load_streaming_datasets(
    dataset_names: Iterable[str],
    split: str,
    probabilities: Optional[List[float]],
    buffer_size: int,
    seed: int,
):
    datasets = [load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=buffer_size) for name in dataset_names]
    if len(datasets) == 1:
        return datasets[0]
    if probabilities:
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("Sum of dataset probabilities must be positive.")
        normalized = [p / total for p in probabilities]
        return interleave_datasets(datasets, probabilities=normalized)
    # Equal interleave avoids loading full dataset while preventing dataset-level ordering bias.
    return interleave_datasets(datasets)


def load_in_memory_datasets(dataset_names: Iterable[str], split: str, probabilities: Optional[List[float]] = None):
    datasets = [load_dataset(name, split=split) for name in dataset_names]
    if len(datasets) == 1:
        return datasets[0]
    if probabilities:
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("Sum of dataset probabilities must be positive.")
        normalized = [p / total for p in probabilities]
        return interleave_datasets(datasets, probabilities=normalized)
    return concatenate_datasets(datasets)


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
    is_streaming = getattr(dataset, "is_streaming", False)
    if is_streaming:
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

    dataset = dataset.map(tokenize_examples, batched=True, remove_columns=dataset.column_names)
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
    if args.streaming:
        dataset = load_streaming_datasets(
            args.dataset_names,
            split="train",
            probabilities=args.dataset_probabilities,
            buffer_size=args.shuffle_buffer_size,
            seed=args.seed + index,
        )
    else:
        dataset = load_in_memory_datasets(args.dataset_names, split="train", probabilities=args.dataset_probabilities)
    dataloader = build_dataloader(dataset, tokenizer, args)

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
