# run_mlm.py
import argparse
import os

import torch
from torch.utils.data import DataLoader

from transformers import (
    DebertaConfig,
    ModernBertForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--tpu_num_cores", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Where to store the tensorboard logs.")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Load dataset
    data_files = {"train": args.train_file}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    raw_datasets = load_dataset("text", data_files=data_files)

    # Load tokenizer
    tokenizer = DebertaTokenizerFast.from_pretrained(args.tokenizer_name_or_path, use_fast=True)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation", None)

    # Create DeBERTa config and model from scratch
    config = DebertaConfig()
    model = DebertaForMaskedLM(config)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Training arguments with tensorboard reporting
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        learning_rate=args.learning_rate,
        report_to=["tensorboard"],  # Enable tensorboard
        logging_dir=args.logging_dir, # Directory for tensorboard logs
        seed=args.seed
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model()

if __name__ == "__main__":
    main()
