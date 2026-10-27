import argparse
import torch
import traceback
import sys
import os
import random
import math
import gc
import datetime
import shutil
import numpy as np
import tempfile
import subprocess
import copy
from time import sleep
from functools import partial
from itertools import chain
import glob
import re
from tqdm import tqdm

import pysbd

from transformers import (
    LlamaTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from datasets import load_dataset, DatasetDict
from safetensors.torch import load_file as load_safetensors

# import wandb
from gcsfs import GCSFileSystem
from google.cloud import storage
import fsspec

# GPU environment setup
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("Number of GPUs available:", torch.cuda.device_count(), flush=True)

def shuffle_and_save_dataset(dataset, save_path):
    """Shuffle and save dataset to disk"""
    print("Shuffling train dataset...")
    shuffled_train = dataset["train"].shuffle(seed=42)

    print("Shuffling validation dataset...")
    shuffled_validation = dataset["validation"].shuffle(seed=42)

    shuffled_dataset = DatasetDict({
        "train": shuffled_train,
        "validation": shuffled_validation
    })

    print(f"Saving shuffled dataset to {save_path}...")
    shuffled_dataset.save_to_disk(save_path)
    print("Dataset shuffled and saved!")


# Global debug flag to limit debug output
_debug_tokenization = True

def _sentence_token_boundaries(text, offsets, segmenter):
    """Map sentence/paragraph endings onto token positions without retokenizing text."""
    # pysbd already locates sentences in the original text. Searching for their
    # strings again can fail for repeated sentences or unusual whitespace.
    character_boundaries = {span.end for span in segmenter.segment(text)}

    for match in re.finditer(r"\n[ \t]*\n+", text):
        character_boundaries.add(match.end())

    boundaries = []
    positions = sorted(character_boundaries)
    position_index = 0
    for token_index, (_, end) in enumerate(offsets):
        if end == 0:  # Special tokens do not correspond to text characters.
            continue
        while position_index < len(positions) and end >= positions[position_index]:
            boundaries.append(token_index + 1)
            position_index += 1
    return sorted(set(boundaries))


def tokenize_function(examples, tokenizer, sentence_aware=False):
    global _debug_tokenization

    # Debug the input (only first batch)
    if _debug_tokenization:
        print(f"DEBUG tokenize: examples keys: {list(examples.keys())}")
        print(f"DEBUG tokenize: number of texts: {len(examples['text'])}")

    # Tokenize the text
    tokenized = tokenizer(
        examples["text"], padding=False, truncation=False,
        return_offsets_mapping=sentence_aware,
    )
    if sentence_aware:
        segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
        tokenized["sentence_boundaries"] = [
            _sentence_token_boundaries(text, offsets, segmenter)
            for text, offsets in zip(examples["text"], tokenized.pop("offset_mapping"))
        ]

    # Debug the output (only first batch)
    if _debug_tokenization:
        print(f"DEBUG tokenize: tokenized keys: {list(tokenized.keys())}")
        print(f"DEBUG tokenize: number of sequences: {len(tokenized['input_ids'])}")
        if len(tokenized["input_ids"]) > 0:
            first_seq = tokenized["input_ids"][0]
            print(f"DEBUG tokenize: first sequence type: {type(first_seq)}")
            print(f"DEBUG tokenize: first sequence length: {len(first_seq)}")

    # Ensure each sample ends with EOS (append only if missing).
    eos_added = 0
    for i in range(len(tokenized["input_ids"])):
        seq = tokenized["input_ids"][i]
        if (len(seq) == 0) or (seq[-1] != tokenizer.eos_token_id):
            tokenized["input_ids"][i].append(tokenizer.eos_token_id)
            if "attention_mask" in tokenized:
                tokenized["attention_mask"][i].append(1)
            eos_added += 1

    if _debug_tokenization:
        if len(tokenized["input_ids"]) > 0:
            print(f"DEBUG tokenize: after EOS, first sequence last token: {tokenized['input_ids'][0][-1]}")
            print(f"DEBUG tokenize: after EOS, first sequence length: {len(tokenized['input_ids'][0])}")
        print(f"DEBUG tokenize: eos_token_id={tokenizer.eos_token_id}, eos_added_in_batch={eos_added}/{len(tokenized['input_ids'])}")
        _debug_tokenization = False  # Disable further debug output

    return tokenized

def load_from_gcs(bucket_name, blob_name, local_path, device):
    """Download file from Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

    # Load the checkpoint
    if local_path.endswith('.safetensors'):
        checkpoint = load_safetensors(local_path, device='cpu')
    else:
        checkpoint = torch.load(local_path, map_location='cpu')

    # Clean up temporary file
    os.remove(local_path)

    return checkpoint

def _parquet_files(path: str, seed: int) -> list[str]:
    files = glob.glob(os.path.join(path, "*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")

    files = sorted(files)

    rng = random.Random(seed)
    rng.shuffle(files)

    return files


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(t) for t in token_ids if t is not None]
    return [int(token_ids)]


def _resolve_eos_token_ids(tokenizer, model=None):
    """Collect a stable set of EOS ids (tokenizer + model config + known Llama EOT/EOS)."""
    eos_ids = []

    eos_ids.extend(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token_name in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = vocab.get(token_name)
        if token_id is not None:
            eos_ids.append(int(token_id))

    if model is not None:
        eos_ids.extend(_normalize_token_ids(getattr(model.config, "eos_token_id", None)))
        if getattr(model, "generation_config", None) is not None:
            eos_ids.extend(
                _normalize_token_ids(getattr(model.generation_config, "eos_token_id", None))
            )

    # Deduplicate while preserving order
    unique_eos_ids = []
    for token_id in eos_ids:
        if token_id not in unique_eos_ids:
            unique_eos_ids.append(token_id)

    return unique_eos_ids


def compute_causal_lm_loss(outputs, labels, eos_token_ids=None, eos_loss_weight=1.0):
    """Compute causal LM loss with optional EOS upweighting."""
    if eos_loss_weight == 1.0 or not eos_token_ids:
        return outputs.loss

    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    token_losses = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    ).view_as(shift_labels)

    valid_mask = shift_labels.ne(-100)
    weights = torch.ones_like(token_losses)

    eos_mask = torch.zeros_like(valid_mask)
    for eos_id in eos_token_ids:
        eos_mask = eos_mask | shift_labels.eq(int(eos_id))

    weights = torch.where(
        eos_mask,
        torch.full_like(weights, float(eos_loss_weight)),
        weights,
    )

    valid_mask_f = valid_mask.to(token_losses.dtype)
    weighted_loss_sum = (token_losses * weights * valid_mask_f).sum()
    weighted_token_count = (weights * valid_mask_f).sum().clamp(min=1.0)

    return weighted_loss_sum / weighted_token_count

def _validate_text_column(dataset):
    """Ensure the train split has the required `text` column."""
    train_split = dataset["train"]
    colnames = getattr(train_split, "column_names", None)

    if colnames:
        if "text" not in colnames:
            raise ValueError(
                "Dataset is missing required 'text' column in train split. "
                f"Found columns: {colnames}"
            )
        return

    # Fallback for iterable datasets where column_names may be unavailable.
    sample = None
    if hasattr(train_split, "take"):
        sample = next(iter(train_split.take(1)), None)

    if sample is None or "text" not in sample:
        raise ValueError(
            "Dataset is missing required 'text' column in train split "
            "and it could not be inferred from a sample."
        )


def load_dataset_from_args(args):
    """Load either regular or streaming train/validation datasets."""

    print(f"Loading dataset from {args.dataset_dir}")

    text_only_columns = ["text"] if args.dataset_format == "parquet" else None
    if text_only_columns is not None:
        print("Parquet mode: loading only the 'text' column")

    if not args.streaming:
        load_kwargs = {
            "data_files": {
                "train": os.path.join(
                    args.dataset_dir,
                    "train",
                    f"*.{args.dataset_format}",
                ),
                "validation": os.path.join(
                    args.dataset_dir,
                    "validation",
                    f"*.{args.dataset_format}",
                ),
            },
            "streaming": False,
        }

        if text_only_columns is not None:
            load_kwargs["columns"] = text_only_columns

        dataset = load_dataset(args.dataset_format, **load_kwargs)
        _validate_text_column(dataset)
        return dataset

    if args.dataset_format != "parquet":
        raise ValueError(
            "--streaming currently expects --dataset_format parquet"
        )

    train_files = _parquet_files(
        os.path.join(args.dataset_dir, "train", "partitions"),
        seed=args.seed,
    )

    validation_files = _parquet_files(
        os.path.join(args.dataset_dir, "validation"),
        seed=args.seed + 1,
    )

    print(
        f"Streaming {len(train_files)} train parquet files and "
        f"{len(validation_files)} validation parquet files"
    )

    print("First train files:")
    for filename in train_files[:5]:
        print(f"  {filename}")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": train_files,
            "validation": validation_files,
        },
        streaming=True,
        columns=["text"],
    )

    _validate_text_column(dataset)

    # Buffered approximate shuffle of examples.
    dataset["train"] = dataset["train"].shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
    )

    return dataset

# Global debug flag for group_texts
_debug_group_texts = True
def keep_min_tokens(example, min_tokens):
    input_ids = example.get("input_ids")
    if input_ids is None:
        return False
    return len(input_ids) >= min_tokens


def group_texts(examples, max_seq_length, pad_token=0):
    """
    Group already tokenized texts into chunks of max_seq_length while respecting sample boundaries.

    Args:
        examples: Dictionary with keys like 'input_ids', 'attention_mask', etc. where each value
                 is a list of tokenized examples
        max_seq_length: Maximum sequence length
        pad_token: Token to use for padding (default: 0)

    Returns:
        Dictionary with chunked input_ids, attention_mask, and labels
    """
    global _debug_group_texts

    # Debug: Print the structure of examples (only first batch)
    if _debug_group_texts:
        print(f"\nDEBUG group_texts: examples keys: {list(examples.keys())}")
        print(f"DEBUG group_texts: type of examples['input_ids']: {type(examples['input_ids'])}")
        print(f"DEBUG group_texts: len of examples['input_ids']: {len(examples['input_ids'])}")
        if len(examples['input_ids']) > 0:
            print(f"DEBUG group_texts: type of first item: {type(examples['input_ids'][0])}")
            if hasattr(examples['input_ids'][0], '__len__'):
                print(f"DEBUG group_texts: length of first item: {len(examples['input_ids'][0])}")
        _debug_group_texts = False  # Disable further debug output

    # Only process input_ids and attention_mask
    keys_to_process = ['input_ids', 'attention_mask']
    result = {k: [] for k in keys_to_process}

    # Loop through each tokenized example
    for i in range(len(examples['input_ids'])):
        # Extract the current tokenized example for input_ids and attention_mask only
        current_example = {k: examples[k][i] for k in keys_to_process if k in examples}

        # Ensure current example values are lists
        for k in current_example:
            if hasattr(current_example[k], 'tolist'):
                current_example[k] = current_example[k].tolist()
            elif not isinstance(current_example[k], list):
                current_example[k] = list(current_example[k])

        example_length = len(current_example['input_ids'])
        boundaries = examples.get('sentence_boundaries', [])[i] if 'sentence_boundaries' in examples else []
        start = 0
        boundary_index = 0
        while start < example_length:
            limit = min(start + max_seq_length, example_length)
            end = limit
            if limit < example_length:
                while boundary_index < len(boundaries) and boundaries[boundary_index] <= start:
                    boundary_index += 1
                next_index = boundary_index
                while next_index < len(boundaries) and boundaries[next_index] <= limit:
                    next_index += 1
                if next_index > boundary_index:
                    end = boundaries[next_index - 1]
                # No complete sentence fits: split the oversized sentence at the limit.
                boundary_index = next_index

            for k, tokens in current_example.items():
                chunk = tokens[start:end]
                if len(chunk) < max_seq_length:
                    padding = [0 if k == 'attention_mask' else pad_token] * (max_seq_length - len(chunk))
                    chunk = chunk + padding
                result[k].append(chunk)
            start = end

    # Add labels for causal LM
    if "input_ids" in result:
        result["labels"] = result["input_ids"].copy()

    return result

def prep_fn(args):
    """Prepare dataset for training."""
    print("Starting dataset preparation")

    dataset = load_dataset_from_args(args)

    print("Tokenizing dataset...")
    tokenize_fn = partial(
        tokenize_function,
        tokenizer=args.tokenizer,
        sentence_aware=args.sentence_aware_chunking,
    )

    candidate_remove = [
        "text",
        "id",
        "source",
        "approx_token_counts_translated",
        "approx_token_counts_original",
    ]

    colnames = dataset["train"].column_names or []
    safe_remove = [
        column
        for column in candidate_remove
        if column in colnames
    ]

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=safe_remove,
    )

    if args.min_tokens_per_sample > 1:
        print(f"Filtering samples with fewer than {args.min_tokens_per_sample} tokens...")
        filter_fn = partial(
            keep_min_tokens,
            min_tokens=args.min_tokens_per_sample,
        )

        if not args.streaming:
            train_before = len(dataset["train"])
            validation_before = len(dataset["validation"])

        dataset = dataset.filter(filter_fn)

        if not args.streaming:
            train_after = len(dataset["train"])
            validation_after = len(dataset["validation"])
            print(
                "Filtered dataset sizes -> "
                f"train: {train_before} -> {train_after}, "
                f"validation: {validation_before} -> {validation_after}"
            )
        else:
            print("Streaming mode: min-token filtering will be applied lazily during iteration")

    print("Grouping texts...")
    group_fn = partial(
        group_texts,
        max_seq_length=args.max_seq_length,
        pad_token=args.tokenizer.pad_token_id,
    )

    dataset = dataset.map(
        group_fn,
        batched=True,
        remove_columns=["sentence_boundaries"] if args.sentence_aware_chunking else None,
    )

    print("Dataset preparation complete")
    return dataset


def safe_iter(dataloader):
    """Safe iterator with error handling"""
    try:
        for batch in dataloader:
            yield batch
    except Exception as e:
        print(f"Error in data iteration: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


def train_fn(tokenized_dataset, device, args):
    """Main training function"""
    print("Starting train_fn...")

    if args.eos_loss_weight <= 0:
        raise ValueError("--eos_loss_weight must be > 0")

    # Initialize wandb
    # wandb.login(key=args.wandb_key)
    # wandb.init(
    #     project="Llama 3.2 - 1B GPU CPT - clinical",
    #     config={
    #         "learning_rate": args.learning_rate,
    #         "architecture": "Llama 3.2 - 1B",
    #         "dataset": args.dataset_dir,
    #         "epochs": args.num_train_epochs,
    #         "weight_decay": args.weight_decay,
    #         "max_seq_length": args.max_seq_length,
    #         "batch_size": args.per_device_train_batch_size,
    #         "bf16": args.bf16,
    #         "precision": "bfloat16" if args.bf16 else "float32",
    #         "vocab_size": len(args.tokenizer),
    #         "pad_token": args.tokenizer.pad_token,
    #         "eos_token": args.tokenizer.eos_token,
    #         "pad_token_id": args.tokenizer.pad_token_id,
    #         "eos_token_id": args.tokenizer.eos_token_id,
    #     },
    #     mode="online",
    #     dir="/tmp"
    # )
    # wandb.run.log_code(root=".", name="cpt_llama_gpu")

    # Load model
    print("Loading the LM...")

    if isinstance(args.checkpoint_path, str) and (args.checkpoint_path != ""):
        print(f"Loading model from checkpoint: {args.checkpoint_path}")

        config = LlamaConfig.from_pretrained(args.model_name, token=args.huggingface_token)
        model = LlamaForCausalLM(config)
        model.config.use_cache = False

        print("Empty model created")

        if args.checkpoint_path.startswith('gs://'):
            # Parse GCS path
            bucket_name = args.checkpoint_path.split('/')[2]
            blob_name = '/'.join(args.checkpoint_path.split('/')[3:])
            local_path = f'/tmp/checkpoint.{args.checkpoint_path.split(".")[-1]}'
            checkpoint = load_from_gcs(bucket_name, blob_name, local_path, device)
            print("Checkpoint downloaded from GCS")
        else:
            if args.checkpoint_path.endswith('.safetensors'):
                checkpoint = load_safetensors(args.checkpoint_path, device='cpu')
            else:
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            print("Checkpoint loaded from local path")

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Checkpoint loaded into model")
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            token=args.huggingface_token,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            low_cpu_mem_usage=True
        )
    model.config.pad_token_id = args.tokenizer.pad_token_id
    eos_token_ids = _resolve_eos_token_ids(args.tokenizer, model=model)
    if not eos_token_ids:
        eos_token_ids = [int(args.tokenizer.eos_token_id)]

    eos_for_config = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
    model.config.eos_token_id = eos_for_config

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = args.tokenizer.pad_token_id
        model.generation_config.eos_token_id = eos_for_config

    print(
        f"Model special token ids -> pad: {model.config.pad_token_id}, "
        f"eos: {model.config.eos_token_id}"
    )
    print(
        f"EOS loss weighting -> eos_ids: {eos_token_ids}, "
        f"eos_loss_weight: {args.eos_loss_weight}"
    )
    if args.eos_loss_weight != 1.0:
        print("Note: loss/perplexity logs are on weighted loss when EOS weighting is enabled")
    print(f"Moving model to device {device}")
    if args.bf16:
        model = model.to(device, dtype=torch.bfloat16)
        print("Model moved to device with bfloat16 precision")
    else:
        model = model.to(device)
        print("Model moved to device with float32 precision")
    print("Model setup complete")

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=args.tokenizer,
        mlm=False
    )

    print("Creating DataLoader...")


    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        shuffle=not args.streaming,
    )
    # Validation dataloader
    validation_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )

    del tokenized_dataset
    gc.collect()

    print("DataLoader created")

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.streaming:
        steps_per_epoch = args.max_steps_per_epoch

        if steps_per_epoch is None or steps_per_epoch <= 0:
            raise ValueError(
                "--max_steps_per_epoch must be > 0 when --streaming is used"
            )
    else:
        steps_per_epoch = len(train_dataloader)

    total_steps = steps_per_epoch * args.num_train_epochs
    save_steps = int(steps_per_epoch * args.save_epoch_percentage)

    if args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=args.num_cycles
        )

    print("Starting training...")
    print(f"Total steps: {total_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Steps before saving: {save_steps}")
    print(f"Total epochs: {args.num_train_epochs}")
    print(f"Warmup steps: {args.num_warmup_steps}")

    # Training loop
    global_step = 0

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Set up automatic mixed precision if bf16 is enabled
    scaler = None
    if args.bf16:
        print("Using bfloat16 automatic mixed precision")
    else:
        print("Using float32 precision")

    for epoch in range(args.num_train_epochs):
        print(f"Starting epoch {epoch}")
        if args.streaming:
            train_dataset = train_dataloader.dataset
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in tqdm(enumerate(safe_iter(train_dataloader)), total=steps_per_epoch):
            if args.streaming and step >= steps_per_epoch:
                break
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward pass with automatic mixed precision
            if args.bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = compute_causal_lm_loss(
                        outputs,
                        labels=batch["labels"],
                        eos_token_ids=eos_token_ids,
                        eos_loss_weight=args.eos_loss_weight,
                    )
            else:
                outputs = model(**batch)
                loss = compute_causal_lm_loss(
                    outputs,
                    labels=batch["labels"],
                    eos_token_ids=eos_token_ids,
                    eos_loss_weight=args.eos_loss_weight,
                )

            nan = torch.isnan(loss)
            if nan:
                optimizer.zero_grad(set_to_none=True)
                print("NaN loss; skipping backward but keeping sync", flush=True)
                loss = torch.nan_to_num(loss, nan=0.0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                #print(f"Clipping grad norm to {args.max_grad_norm}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Logging
            if step % args.logging_steps == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                perplexity = math.exp(avg_loss)

                print(f"Epoch {epoch}, Step {step}, Global Step {global_step}")
                print(f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
                print(f"Learning Rate: {current_lr:.2e}")

                # if wandb.run is not None:
                #     wandb.log({
                #         "train/loss": loss.item(),
                #         "train/avg_loss": avg_loss,
                #         "train/perplexity": perplexity,
                #         "train/learning_rate": current_lr,
                #         "train/epoch": epoch,
                #         "train/global_step": global_step,
                #     })


            # Debug mode - only run a few steps
            if args.debug and step >= 10:
                break

            if (step % save_steps == 0) and (step > 0):
                print("Running validation...")
                val_results = evaluate(model, validation_dataloader, device, args, eos_token_ids)
                print(f"Validation Loss: {val_results['eval_loss']:.4f}, Perplexity: {val_results['perplexity']:.4f}")

                print("Saving model...")
                model_cpu = copy.deepcopy(model).to("cpu")
                model_cpu.save_pretrained(args.output_dir, save_serialization=True)

        # Evaluation at end of epoch
        print("Running validation...")
        val_results = evaluate(model, validation_dataloader, device, args, eos_token_ids)
        print(f"Validation Loss: {val_results['eval_loss']:.4f}, Perplexity: {val_results['perplexity']:.4f}")

        # if wandb.run is not None:
        #     wandb.log({
        #         "eval/loss": val_results['eval_loss'],
        #         "eval/perplexity": val_results['perplexity'],
        #         "epoch": epoch,
        #     })

        # Save checkpoint at end of epoch
        checkpoint_path = f"{args.output_dir}/checkpoint-epoch-{epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"Saving checkpoint to {checkpoint_path}")

        # Save model
        model.save_pretrained(checkpoint_path)
        args.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'avg_loss': epoch_loss / num_batches,
        }, f"{checkpoint_path}/training_state.pt")

        print(f"Checkpoint saved to {checkpoint_path}")

        # Upload to GCS if output_dir is a GCS path
        if args.output_dir.startswith("gs://"):
            print(f"Uploading checkpoint to GCS: {args.output_dir}")
            subprocess.run(["gsutil", "-m", "cp", "-r", checkpoint_path, args.output_dir], check=True)

    print("Training completed!")
    #wandb.finish()


def evaluate(model, validation_dataloader, device, args, eos_token_ids):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in validation_dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if args.bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = compute_causal_lm_loss(
                        outputs,
                        labels=batch["labels"],
                        eos_token_ids=eos_token_ids,
                        eos_loss_weight=args.eos_loss_weight,
                    )
            else:
                outputs = model(**batch)
                loss = compute_causal_lm_loss(
                    outputs,
                    labels=batch["labels"],
                    eos_token_ids=eos_token_ids,
                    eos_loss_weight=args.eos_loss_weight,
                )

            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)

    model.train()  # Switch back to train mode
    return {"eval_loss": avg_loss, "perplexity": perplexity}


def prep_and_train_fn(args):
    """Setup training and run training"""
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    print(f"Using device {device}")

    # Prepare dataset
    tokenized_dataset = prep_fn(args)

    # Start training
    train_fn(tokenized_dataset, device, args)


def main():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_format", default="json", choices=["json", "parquet", "arrow"])
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pre_tokenized", action='store_true', default=False)

    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument(
        "--sentence_aware_chunking", action="store_true",
        help="End chunks at sentence/paragraph boundaries when possible; split oversized sentences.",
    )

    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--lr_schedule", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--num_cycles", type=float, default=10)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=21.0)
    parser.add_argument(
        "--eos_loss_weight",
        type=float,
        default=1.0,
        help="Optional loss weight applied to EOS-token targets (>1 upweights EOS).",
    )

    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_epoch_percentage", type=float, default=1.0)

    # Data processing arguments
    parser.add_argument("--keep_in_memory", action='store_true')
    parser.add_argument(
        "--min_tokens_per_sample",
        type=int,
        default=16,
        help="Minimum number of tokens per sample before chunking (after EOS append).",
    )
    parser.add_argument("--max_steps_per_epoch", type=int, default=250_000, help="Required with --streaming; optimizer steps per epoch.",)

    # System arguments
    parser.add_argument("--seed", type=int, default=42)

    # External service arguments
    parser.add_argument("--wandb_key", type=str, required=True)
    parser.add_argument("--huggingface_token", type=str, default=None)

    # Debug and precision
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--bf16", action='store_true', help="Use bfloat16 precision for training")

    parser.add_argument(
    "--streaming",
    action="store_true",
    help=(
        "Stream parquet files instead of materializing the complete dataset. "
        "Training parquet files are randomized and examples are shuffled "
        "using a bounded shuffle buffer."
        ),
    )

    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="Number of examples held in the streaming shuffle buffer.",
    )

    args = parser.parse_args()

    if args.max_seq_length < 1:
        raise ValueError("--max_seq_length must be >= 1")
    if args.min_tokens_per_sample < 1:
        raise ValueError("--min_tokens_per_sample must be >= 1")

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize tokenizer
    args.tokenizer = LlamaTokenizerFast.from_pretrained(
        args.tokenizer_name_or_path,
        token=args.huggingface_token
    )
    args.tokenizer.model_max_length = args.max_seq_length
    args.tokenizer.pad_token = "<|finetune_right_pad_id|>"

    assert args.tokenizer.pad_token_id == 128004
    assert args.tokenizer.eos_token_id == 128009
    assert args.tokenizer.pad_token_id != args.tokenizer.eos_token_id

    # Ensure EOS token is properly set
    if args.tokenizer.eos_token is None:
        print("Warning: No EOS token found in tokenizer, using default", flush=True)
        args.tokenizer.add_special_tokens({'eos_token': '<|eot_id|>'})
    if args.tokenizer.bos_token is None:
        print("Warning: No BOS token found in tokenizer, using default", flush=True)
        args.tokenizer.add_special_tokens({'bos_token': '<|begin_of_text|>'})

    print(f"Tokenizer EOS token: {args.tokenizer.eos_token} (ID: {args.tokenizer.eos_token_id})")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting single GPU training...")
    prep_and_train_fn(args)
    print("Training completed!")


if __name__ == "__main__":
    main()
