#!/usr/bin/env python3
"""Train or continue-pretrain a DeBERTa-v2 masked-language model.

Key properties
--------------
* Packs multiple documents into predominantly full token windows.
* Inserts a separator token between documents.
* Adds sequence-level special tokens only after packing.
* Uses DataCollatorForLanguageModeling for dynamic padding and MLM masking.
* Recreates deterministic validation masks for every evaluation.
* Computes scheduler steps in optimizer updates, not microbatches.
* Handles the final partial gradient-accumulation update correctly.
* Supports regular and streaming Hugging Face datasets.

Expected raw dataset columns
----------------------------
At minimum, each raw Parquet split must contain a ``text`` column. Other
columns are removed automatically.

Pre-tokenized mode
------------------
With ``--pre_tokenized``, the script loads:

    <dataset_dir>/train_<max_seq_length>.parquet
    <dataset_dir>/validation_<max_seq_length>.parquet

The loaded token sequences are stripped of padding and sequence-level special
tokens, and are then repacked. This means legacy pre-tokenized files created by
the previous script can still benefit from the new packing strategy.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import inspect
import math
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import ftfy
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2TokenizerFast,
    get_linear_schedule_with_warmup,
)


RE_SPURIOUS_CHARS = re.compile(r"([^\w])\1{3,}")
RE_SPURIOUS_WORDS = re.compile(r"(\b[\w\-\s\;\:\,\.]+\b)\1{4,}")
RE_MULTISPACE = re.compile(r"\s{2,}")

# Architectural context limit. --max_seq_length controls the shorter
# training windows used during a particular pretraining stage.
MODEL_MAX_POSITION_EMBEDDINGS = 1024


def apply_until_stable(
    pattern: re.Pattern[str],
    repl: str,
    text: str,
    max_iter: int = 20,
) -> str:
    for _ in range(max_iter):
        text, changed = pattern.subn(repl, text)
        if changed == 0:
            break
    return text


def clean_text(text: Any, num_reps: int = 20) -> str:
    if text is None:
        return ""

    text = str(text)
    text = apply_until_stable(RE_SPURIOUS_WORDS, r"\1", text, num_reps)
    text = RE_SPURIOUS_CHARS.sub(r"\1", text)
    text = apply_until_stable(RE_SPURIOUS_WORDS, r"\1", text, num_reps)
    text = RE_MULTISPACE.sub(" ", text)
    return ftfy.fix_encoding(text).strip()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker distinct Python and NumPy RNG states."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_separator_token_id(tokenizer: DebertaV2TokenizerFast) -> Optional[int]:
    for token_id in (
        tokenizer.sep_token_id,
        tokenizer.eos_token_id,
    ):
        if token_id is not None:
            return int(token_id)
    return None


def strip_padding_and_special_tokens(
    input_ids: Sequence[int],
    tokenizer: DebertaV2TokenizerFast,
    attention_mask: Optional[Sequence[int]] = None,
) -> List[int]:
    """Recover document content tokens from a pre-tokenized sequence."""
    ids = list(map(int, input_ids))

    if attention_mask is not None:
        mask = list(map(int, attention_mask))
        ids = [token_id for token_id, keep in zip(ids, mask) if keep != 0]
    elif tokenizer.pad_token_id is not None:
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()

    if not ids:
        return []

    special_mask = tokenizer.get_special_tokens_mask(
        ids,
        already_has_special_tokens=True,
    )
    return [
        token_id
        for token_id, is_special in zip(ids, special_mask)
        if not is_special
    ]


def pack_documents(
    documents: Iterable[Sequence[int]],
    tokenizer: DebertaV2TokenizerFast,
    max_seq_length: int,
    keep_remainder: bool = True,
) -> Dict[str, List[List[int]]]:
    """Pack tokenized documents into near-full MLM sequences.

    A separator is inserted between documents. Sequence-level special tokens
    such as [CLS] and [SEP] are added only after a content chunk is formed.

    The function is intended for ``Dataset.map(..., batched=True)`` with a
    large ``batch_size``. It emits at most one incomplete sequence per mapping
    batch, so packing loss is negligible when ``packing_batch_size`` is large.
    """
    num_outer_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_capacity = max_seq_length - num_outer_special_tokens
    if content_capacity <= 0:
        raise ValueError(
            "max_seq_length is too small for the tokenizer's required special tokens: "
            f"max_seq_length={max_seq_length}, "
            f"special_tokens={num_outer_special_tokens}."
        )

    separator_id = get_separator_token_id(tokenizer)
    token_stream: List[int] = []

    for document in documents:
        content = list(map(int, document))
        if not content:
            continue

        token_stream.extend(content)
        if separator_id is not None:
            token_stream.append(separator_id)

    result: Dict[str, List[List[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "special_tokens_mask": [],
    }

    if not token_stream:
        return result

    full_length = (len(token_stream) // content_capacity) * content_capacity
    stop = len(token_stream) if keep_remainder else full_length

    for start in range(0, stop, content_capacity):
        content_chunk = token_stream[start : start + content_capacity]
        if not content_chunk:
            continue
        if len(content_chunk) < content_capacity and not keep_remainder:
            break

        input_ids = tokenizer.build_inputs_with_special_tokens(content_chunk)
        if len(input_ids) > max_seq_length:
            raise RuntimeError(
                "Packed sequence exceeds max_seq_length: "
                f"{len(input_ids)} > {max_seq_length}."
            )

        special_tokens_mask = tokenizer.get_special_tokens_mask(
            input_ids,
            already_has_special_tokens=True,
        )

        result["input_ids"].append(input_ids)
        result["attention_mask"].append([1] * len(input_ids))
        result["special_tokens_mask"].append(special_tokens_mask)

    return result


def make_raw_pack_function(
    tokenizer: DebertaV2TokenizerFast,
    max_seq_length: int,
):
    def tokenize_and_pack(examples: Mapping[str, Sequence[Any]]) -> Dict[str, List[List[int]]]:
        if "text" not in examples:
            raise KeyError("Raw datasets must contain a 'text' column.")

        texts = [clean_text(text) for text in examples["text"]]
        texts = [text for text in texts if text]
        if not texts:
            return {
                "input_ids": [],
                "attention_mask": [],
                "special_tokens_mask": [],
            }

        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        return pack_documents(
            documents=tokenized["input_ids"],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            keep_remainder=True,
        )

    return tokenize_and_pack


def make_pretokenized_repack_function(
    tokenizer: DebertaV2TokenizerFast,
    max_seq_length: int,
):
    def repack(examples: Mapping[str, Sequence[Any]]) -> Dict[str, List[List[int]]]:
        if "input_ids" not in examples:
            raise KeyError("Pre-tokenized datasets must contain an 'input_ids' column.")

        attention_masks = examples.get("attention_mask")
        documents: List[List[int]] = []

        for index, input_ids in enumerate(examples["input_ids"]):
            attention_mask = (
                attention_masks[index] if attention_masks is not None else None
            )
            document = strip_padding_and_special_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
            )
            if document:
                documents.append(document)

        return pack_documents(
            documents=documents,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            keep_remainder=True,
        )

    return repack


def process_split(
    split_dataset: Any,
    transform: Any,
    args: argparse.Namespace,
    split_name: str,
) -> Any:
    map_kwargs: Dict[str, Any] = {
        "batched": True,
        "batch_size": args.packing_batch_size,
    }
    if split_dataset.column_names:
        map_kwargs["remove_columns"] = split_dataset.column_names

    if not args.streaming_data:
        map_kwargs.update(
            {
                "num_proc": args.num_cores,
                "keep_in_memory": args.keep_in_memory,
                "desc": (
                    f"Tokenizing and packing {split_name} into windows of "
                    f"{args.max_seq_length} tokens"
                ),
            }
        )

    return split_dataset.map(transform, **map_kwargs)


def prepare_datasets(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], DebertaV2TokenizerFast]:
    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        args.tokenizer_name_or_path,
    )
    tokenizer.model_max_length = args.max_seq_length

    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad token.")
    if tokenizer.mask_token_id is None:
        raise ValueError("The tokenizer must define a mask token for MLM training.")

    if args.pre_tokenized:
        print("Loading and repacking pre-tokenized datasets...", flush=True)
        data_files = {
            "train": str(
                Path(args.dataset_dir) / f"train_{args.max_seq_length}.parquet"
            ),
            "validation": str(
                Path(args.dataset_dir)
                / f"validation_{args.max_seq_length}.parquet"
            ),
        }
        raw_datasets = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=args.streaming_data,
            keep_in_memory=args.keep_in_memory,
        )
        transform = make_pretokenized_repack_function(
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        print(
            "Loading, cleaning, tokenizing, and packing raw datasets "
            f"(streaming={args.streaming_data}, "
            f"keep_in_memory={args.keep_in_memory})...",
            flush=True,
        )
        base = Path(args.dataset_dir)
        train_dir = base / "train" / "partitions"
        train_parquets = sorted(train_dir.glob("*.parquet"))
        if not train_parquets:
            raise FileNotFoundError(
                f"No training Parquet files found in {train_dir}."
            )

        file_rng = random.Random(args.seed)
        file_rng.shuffle(train_parquets)

        validation_files = sorted((base / "validation").glob("*.parquet"))
        if not validation_files:
            raise FileNotFoundError(
                f"No validation Parquet files found in {base / 'validation'}."
            )

        data_files = {
            "train": [str(path) for path in train_parquets],
            "validation": [str(path) for path in validation_files],
        }
        raw_datasets = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=args.streaming_data,
            keep_in_memory=args.keep_in_memory,
        )
        transform = make_raw_pack_function(
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )

    packed_datasets = {
        split_name: process_split(
            split_dataset=split_dataset,
            transform=transform,
            args=args,
            split_name=split_name,
        )
        for split_name, split_dataset in raw_datasets.items()
    }

    if args.streaming_data:
        packed_datasets["train"] = packed_datasets["train"].shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer_size,
        )

    return packed_datasets, tokenizer


def build_mlm_collator(
    tokenizer: DebertaV2TokenizerFast,
    mlm_probability: float,
    pad_to_multiple_of: Optional[int],
    seed: Optional[int] = None,
) -> DataCollatorForLanguageModeling:
    kwargs: Dict[str, Any] = {
        "tokenizer": tokenizer,
        "mlm": True,
        "mlm_probability": mlm_probability,
        "pad_to_multiple_of": pad_to_multiple_of,
        "return_tensors": "pt",
    }

    # Newer Transformers releases support an independent masking RNG. Retain
    # compatibility with older releases by adding this argument conditionally.
    if (
        seed is not None
        and "seed"
        in inspect.signature(DataCollatorForLanguageModeling).parameters
    ):
        kwargs["seed"] = seed

    return DataCollatorForLanguageModeling(**kwargs)


def get_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "layernorm.weight",
    )

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": args.weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def build_model(
    args: argparse.Namespace,
    tokenizer: DebertaV2TokenizerFast,
    device: torch.device,
) -> torch.nn.Module:
    if args.init_training:
        print("Initializing a new DeBERTa-v2 MLM from random weights.", flush=True)
        config = DebertaV2Config(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            pooler_hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=MODEL_MAX_POSITION_EMBEDDINGS,
            norm_rel_ebd="layer_norm",
            relative_attention=True,
            legacy=False,
            pos_att_type=["c2p", "p2c"],
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
        )
        model = AutoModelForMaskedLM.from_config(config)
    else:
        print(
            f"Continuing MLM pretraining from {args.model_name}.",
            flush=True,
        )
        config = DebertaV2Config.from_pretrained(args.model_name)
        config.legacy = False

        model = DebertaV2ForMaskedLM.from_pretrained(
            args.model_name,
            config=config,
        )
        model_max_positions = getattr(
            model.config,
            "max_position_embeddings",
            MODEL_MAX_POSITION_EMBEDDINGS,
        )
        if model_max_positions < args.max_seq_length:
            raise ValueError(
                f"Loaded model supports max_position_embeddings={model_max_positions}, "
                f"but --max_seq_length={args.max_seq_length}. Use a compatible "
                "checkpoint or reduce --max_seq_length."
            )
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            print(
                "Resizing token embeddings from "
                f"{model.get_input_embeddings().num_embeddings} to {len(tokenizer)}.",
                flush=True,
            )
            model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model.config.use_cache = False
    model.to(device)
    return model


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: DebertaV2TokenizerFast,
    output_dir: str,
    checkpoint_name: str,
    tmp_dir: str,
) -> None:
    print(f"Saving {checkpoint_name}...", flush=True)

    if output_dir.startswith("gs://"):
        os.makedirs(tmp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_dir) as temporary_directory:
            local_checkpoint = Path(temporary_directory) / checkpoint_name
            local_checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(local_checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(local_checkpoint)

            destination = output_dir.rstrip("/") + "/"
            subprocess.run(
                [
                    "gsutil",
                    "-m",
                    "cp",
                    "-r",
                    str(local_checkpoint),
                    destination,
                ],
                check=True,
            )
    else:
        local_checkpoint = Path(output_dir) / checkpoint_name
        local_checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(local_checkpoint)


def make_autocast_context(
    device: torch.device,
    use_bf16: bool,
):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return contextlib.nullcontext()


def preserve_rng_state() -> Tuple[
    object,
    tuple,
    torch.Tensor,
    Optional[List[torch.Tensor]],
]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    return python_state, numpy_state, cpu_state, cuda_state


def restore_rng_state(
    state: Tuple[
        object,
        tuple,
        torch.Tensor,
        Optional[List[torch.Tensor]],
    ],
) -> None:
    python_state, numpy_state, cpu_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    validation_dataset: Any,
    tokenizer: DebertaV2TokenizerFast,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, int]:
    """Evaluate with the same random masking pattern on every invocation."""
    rng_state = preserve_rng_state()
    set_global_seed(args.validation_mask_seed)

    try:
        validation_collator = build_mlm_collator(
            tokenizer=tokenizer,
            mlm_probability=args.mlm_proba,
            pad_to_multiple_of=args.pad_to_multiple_of,
            seed=args.validation_mask_seed,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=validation_collator,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model.eval()
        total_negative_log_likelihood = 0.0
        total_masked_targets = 0

        for batch in validation_loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            masked_targets = int(batch["labels"].ne(-100).sum().item())
            if masked_targets == 0:
                continue

            with make_autocast_context(device, args.bf16):
                outputs = model(**batch)

            total_negative_log_likelihood += (
                float(outputs.loss.item()) * masked_targets
            )
            total_masked_targets += masked_targets

        if total_masked_targets == 0:
            raise RuntimeError("Validation produced zero masked target tokens.")

        average_loss = total_negative_log_likelihood / total_masked_targets
        perplexity = math.exp(min(average_loss, 50.0))
        return perplexity, average_loss, total_masked_targets
    finally:
        restore_rng_state(rng_state)
        model.train()


def inspect_first_batch(
    train_dataset: Any,
    collator: DataCollatorForLanguageModeling,
    args: argparse.Namespace,
) -> None:
    """Print one-time token-utilization diagnostics without consuming training RNG."""
    rng_state = preserve_rng_state()
    set_global_seed(args.seed + 12345)
    try:
        diagnostic_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(diagnostic_loader))

        allocated_tokens = int(batch["input_ids"].numel())
        real_tokens = int(batch["attention_mask"].sum().item())
        masked_targets = int(batch["labels"].ne(-100).sum().item())
        padding_fraction = 1.0 - (real_tokens / max(allocated_tokens, 1))

        pad_positions = batch["input_ids"].eq(args.tokenizer.pad_token_id)
        if pad_positions.any():
            if not torch.all(batch["attention_mask"][pad_positions] == 0):
                raise AssertionError(
                    "Padded token positions have a nonzero attention mask."
                )
            if not torch.all(batch["labels"][pad_positions] == -100):
                raise AssertionError(
                    "Padded token positions are not ignored in MLM labels."
                )

        print(
            "First-batch diagnostics: "
            f"shape={tuple(batch['input_ids'].shape)}, "
            f"real_tokens={real_tokens:,}, "
            f"masked_targets={masked_targets:,}, "
            f"padding_fraction={padding_fraction:.2%}",
            flush=True,
        )
    finally:
        restore_rng_state(rng_state)


def divide_gradients(model: torch.nn.Module, divisor: int) -> None:
    if divisor <= 0:
        raise ValueError("Gradient divisor must be positive.")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.", flush=True)

    if args.bf16:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            print(
                "BF16 is unavailable on this device; falling back to FP32.",
                flush=True,
            )
            args.bf16 = False
        else:
            print("Using CUDA BF16 autocast.", flush=True)
    else:
        print("Using FP32 training.", flush=True)

    packed_datasets, tokenizer = prepare_datasets(args)
    args.tokenizer = tokenizer

    train_dataset = packed_datasets["train"]
    validation_dataset = packed_datasets["validation"]
    del packed_datasets
    gc.collect()

    model = build_model(args, tokenizer, device)

    training_collator = build_mlm_collator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_proba,
        pad_to_multiple_of=args.pad_to_multiple_of,
        seed=None,
    )

    inspect_first_batch(
        train_dataset=train_dataset,
        collator=training_collator,
        args=args,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": args.per_device_train_batch_size,
        "collate_fn": training_collator,
        "num_workers": 0 if args.streaming_data else args.dataloader_num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }

    if args.streaming_data:
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True
        train_loader_kwargs["generator"] = loader_generator
        if args.dataloader_num_workers > 0:
            train_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(**train_loader_kwargs)

    if args.streaming_data:
        microbatches_per_epoch = args.max_steps_per_epoch
    else:
        microbatches_per_epoch = len(train_loader)
        if args.max_steps_per_epoch is not None:
            microbatches_per_epoch = min(
                microbatches_per_epoch,
                args.max_steps_per_epoch,
            )

    optimizer_updates_per_epoch = math.ceil(
        microbatches_per_epoch / args.gradient_accumulation_steps
    )
    total_optimizer_updates = (
        optimizer_updates_per_epoch * args.num_train_epochs
    )

    if args.num_warmup_steps >= total_optimizer_updates:
        print(
            "Warning: num_warmup_steps is greater than or equal to the total "
            "number of optimizer updates. The learning rate will remain in "
            "warmup for the full run.",
            flush=True,
        )

    optimizer = get_optimizer(model, args)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_optimizer_updates,
    )

    save_interval = 0
    if args.save_epoch_percentage > 0:
        save_interval = max(
            1,
            round(
                optimizer_updates_per_epoch
                * args.save_epoch_percentage
            ),
        )

    print("Starting training...", flush=True)
    print(f"Epochs: {args.num_train_epochs}", flush=True)
    print(
        f"Microbatches per epoch: {microbatches_per_epoch:,}",
        flush=True,
    )
    print(
        f"Optimizer updates per epoch: {optimizer_updates_per_epoch:,}",
        flush=True,
    )
    print(
        f"Total optimizer updates: {total_optimizer_updates:,}",
        flush=True,
    )
    print(f"Warmup optimizer updates: {args.num_warmup_steps:,}", flush=True)
    print(
        "Approximate full-window tokens per optimizer update: "
        f"{args.per_device_train_batch_size * args.max_seq_length * args.gradient_accumulation_steps:,}",
        flush=True,
    )

    global_microbatch_step = 0
    global_optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_train_epochs):
        if args.streaming_data and hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        model.train()
        accumulation_count = 0
        optimizer_step_in_epoch = 0

        epoch_nll = 0.0
        epoch_targets = 0
        window_nll = 0.0
        window_targets = 0
        window_real_tokens = 0
        window_allocated_tokens = 0

        for microbatch_index, batch in enumerate(train_loader):
            if microbatch_index >= microbatches_per_epoch:
                break

            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            masked_targets = int(batch["labels"].ne(-100).sum().item())
            if masked_targets == 0:
                print(
                    f"Skipping microbatch {microbatch_index}: no MLM targets.",
                    flush=True,
                )
                continue

            with make_autocast_context(device, args.bf16):
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                print(
                    f"Skipping non-finite loss at epoch={epoch + 1}, "
                    f"microbatch={microbatch_index + 1}: {loss.item()}",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                continue

            # Accumulate unnormalized microbatch gradients. Immediately before
            # each optimizer step, divide by the actual number of accumulated
            # microbatches. This also handles a short final accumulation group.
            loss.backward()
            accumulation_count += 1

            batch_nll = float(loss.item()) * masked_targets
            epoch_nll += batch_nll
            epoch_targets += masked_targets
            window_nll += batch_nll
            window_targets += masked_targets
            window_real_tokens += int(batch["attention_mask"].sum().item())
            window_allocated_tokens += int(batch["input_ids"].numel())

            global_microbatch_step += 1

            is_last_microbatch = (
                microbatch_index + 1 >= microbatches_per_epoch
            )
            should_update = (
                accumulation_count >= args.gradient_accumulation_steps
                or is_last_microbatch
            )

            if should_update:
                divide_gradients(model, accumulation_count)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                accumulation_count = 0
                optimizer_step_in_epoch += 1
                global_optimizer_step += 1

                if (
                    save_interval > 0
                    and optimizer_step_in_epoch % save_interval == 0
                    and optimizer_step_in_epoch < optimizer_updates_per_epoch
                ):
                    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                    save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir,
                        checkpoint_name=(
                            f"checkpoint-epoch-{epoch + 1}-"
                            f"update-{optimizer_step_in_epoch}-{timestamp}"
                        ),
                        tmp_dir=args.tmp_dir,
                    )

            if (
                global_microbatch_step % args.logging_steps == 0
                and window_targets > 0
            ):
                window_loss = window_nll / window_targets
                window_perplexity = math.exp(min(window_loss, 50.0))
                padding_fraction = 1.0 - (
                    window_real_tokens / max(window_allocated_tokens, 1)
                )
                current_lr = scheduler.get_last_lr()[0]
                grad_norm_value = (
                    float(grad_norm)
                    if "grad_norm" in locals()
                    else float("nan")
                )

                print(
                    f"epoch={epoch + 1} "
                    f"microbatch={microbatch_index + 1}/{microbatches_per_epoch} "
                    f"optimizer_update={global_optimizer_step}/{total_optimizer_updates} "
                    f"loss={window_loss:.6f} "
                    f"ppl={window_perplexity:.3f} "
                    f"masked_targets={window_targets:,} "
                    f"padding={padding_fraction:.2%} "
                    f"lr={current_lr:.3e} "
                    f"grad_norm={grad_norm_value:.4f}",
                    flush=True,
                )

                window_nll = 0.0
                window_targets = 0
                window_real_tokens = 0
                window_allocated_tokens = 0

        if accumulation_count > 0:
            # This is mainly a safeguard for a streaming iterator that ends
            # before max_steps_per_epoch.
            divide_gradients(model, accumulation_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_optimizer_step += 1
            optimizer_step_in_epoch += 1

        if epoch_targets == 0:
            raise RuntimeError("Training epoch produced zero masked target tokens.")

        epoch_loss = epoch_nll / epoch_targets
        epoch_perplexity = math.exp(min(epoch_loss, 50.0))
        print(
            f"Epoch {epoch + 1} training loss={epoch_loss:.6f}, "
            f"perplexity={epoch_perplexity:.3f}, "
            f"masked_targets={epoch_targets:,}",
            flush=True,
        )

        validation_perplexity, validation_loss, validation_targets = evaluate(
            model=model,
            validation_dataset=validation_dataset,
            tokenizer=tokenizer,
            args=args,
            device=device,
        )
        print(
            f"Epoch {epoch + 1} validation loss={validation_loss:.6f}, "
            f"perplexity={validation_perplexity:.3f}, "
            f"masked_targets={validation_targets:,}",
            flush=True,
        )

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            checkpoint_name=f"checkpoint-epoch-{epoch + 1}-{timestamp}",
            tmp_dir=args.tmp_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or continue-pretrain a packed DeBERTa-v2 MLM."
    )

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="UMCU/CardioDeberta.nl")

    parser.add_argument("--pre_tokenized", action="store_true")
    parser.add_argument("--streaming_data", action="store_true")
    parser.add_argument("--keep_in_memory", action="store_true")
    parser.add_argument(
        "--sharded_data",
        action="store_true",
        help=(
            "Retained for command-line compatibility. Single-process training "
            "does not require an additional one-shard operation."
        ),
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help=(
            "Packed sequence length used for this pretraining stage. "
            "Must be <= 1024. Use 512 for the main pretraining stage and "
            "1024 for long-context continuation."
        ),
    )
    parser.add_argument(
        "--packing_batch_size",
        type=int,
        default=10_000,
        help=(
            "Number of documents concatenated per Dataset.map call. At most "
            "one incomplete packed sequence is emitted per mapping batch."
        ),
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=8,
        help="Dynamic padding multiple used by the MLM data collator.",
    )
    parser.add_argument("--mlm_proba", type=float, default=0.15)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=None,
        help=(
            "Maximum number of microbatches per epoch. Required for streaming "
            "datasets, because their DataLoader has no length."
        ),
    )

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument(
        "--save_epoch_percentage",
        type=float,
        default=0.5,
        help=(
            "Save an intermediate checkpoint after this fraction of each "
            "epoch. Set to 0 to save only at epoch boundaries."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_mask_seed", type=int, default=1042)
    parser.add_argument("--num_cores", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--shuffle_buffer_size", type=int, default=25_000)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--init_training",
        action="store_true",
        help="Initialize a new random model instead of loading model_name.",
    )
    parser.add_argument("--hidden_size", type=int, default=1152)
    parser.add_argument("--num_hidden_layers", type=int, default=24)
    parser.add_argument("--num_attention_heads", type=int, default=24)
    parser.add_argument("--intermediate_size", type=int, default=4608)

    args = parser.parse_args()

    if args.keep_in_memory and args.streaming_data:
        parser.error("--keep_in_memory and --streaming_data are mutually exclusive.")
    if args.streaming_data and args.max_steps_per_epoch is None:
        parser.error("--max_steps_per_epoch is required with --streaming_data.")
    if args.streaming_data and args.shuffle_buffer_size <= 0:
        parser.error("--shuffle_buffer_size must be positive.")
    if args.max_seq_length < 2:
        parser.error("--max_seq_length must be at least 2.")
    if args.max_seq_length > MODEL_MAX_POSITION_EMBEDDINGS:
        parser.error(
            f"--max_seq_length={args.max_seq_length} exceeds the model's "
            f"max_position_embeddings={MODEL_MAX_POSITION_EMBEDDINGS}."
        )
    if args.packing_batch_size <= 0:
        parser.error("--packing_batch_size must be positive.")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size must be positive.")
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    if args.per_device_eval_batch_size <= 0:
        parser.error("--per_device_eval_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive.")
    if args.num_train_epochs <= 0:
        parser.error("--num_train_epochs must be positive.")
    if not 0.0 < args.mlm_proba < 1.0:
        parser.error("--mlm_proba must be between 0 and 1.")
    if args.pad_to_multiple_of is not None and args.pad_to_multiple_of <= 0:
        parser.error("--pad_to_multiple_of must be positive.")
    if args.logging_steps <= 0:
        parser.error("--logging_steps must be positive.")
    if not 0.0 <= args.save_epoch_percentage <= 1.0:
        parser.error("--save_epoch_percentage must be in [0, 1].")
    if args.num_attention_heads <= 0:
        parser.error("--num_attention_heads must be positive.")
    if args.hidden_size % args.num_attention_heads != 0:
        parser.error("--hidden_size must be divisible by --num_attention_heads.")

    return args


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    print(
        f"Training sequence length: {args.max_seq_length} "
        f"(architectural maximum: {MODEL_MAX_POSITION_EMBEDDINGS})",
        flush=True,
    )

    if args.sharded_data:
        print(
            "Note: --sharded_data is ignored in this single-process script; "
            "the dataset is already consumed by one process.",
            flush=True,
        )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train(args)


if __name__ == "__main__":
    print("Starting packed DeBERTa MLM training...", flush=True)
    main()