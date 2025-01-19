#!/usr/bin/env python
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import pysbd
import tqdm

# Create an argument parser to accept file paths, tokenizer model, and max_seq_length.
argparser = argparse.ArgumentParser(description="Preprocess and chunk dataset documents.")
argparser.add_argument("--train_loc", type=str, required=True, help="Path to the training JSON file.")
argparser.add_argument("--validation_loc", type=str, required=True, help="Path to the validation JSON file.")
argparser.add_argument("--save_dir", type=str, required=True, help="Directory where the preprocessed dataset will be saved.")
argparser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Pretrained tokenizer model name or path (e.g., 'microsoft/deberta-v3-base').")
argparser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length for each chunk.")

args = argparser.parse_args()


print(f"Loading tokenizer from {args.tokenizer_name_or_path}...")
_tokenizer = ByteLevelBPETokenizer.from_file(merges_filename=os.path.join(args.tokenizer_name_or_path, 'merges.txt'),
    vocab_filename=os.path.join(args.tokenizer_name_or_path, 'vocab.json'))

print(f"Casting tokenizer TokenizerFast...")
tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tokenizer._tokenizer, truncation=True, model_max_length=args.max_seq_length)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.add_special_tokens({'bos_token': '<s>'})
tokenizer.add_special_tokens({'eos_token': '</s>'})
tokenizer.add_special_tokens({'unk_token': '<unk>'})
tokenizer.add_special_tokens({'mask_token': '<mask>'})

print("Tokenizer loaded.")

# Define data_files dictionary for the dataset loader.
data_files = {
                "train": args.train_loc,
                "validation": args.validation_loc
}

print("Loading dataset...")
raw_datasets = load_dataset("json", data_files=data_files, keep_in_memory=True, num_proc=1)
print("Dataset loaded.")

segmenter = pysbd.Segmenter(language="nl", clean=True)

def chunk_text(example):
    """
    This function splits the full text in example["text"] into sentences using pysbd,
    then iteratively adds sentences until the tokenized length exceeds args.max_seq_length.
    Each chunk is returned as a dict containing the original text chunk (under "text")
    as well as the corresponding "input_ids" (which are tokenized and truncated to args.max_seq_length).
    """
    full_text = example["text"]

    # Split the full text into sentences
    try:
        sentences = segmenter.segment(full_text)

        chunks = []         # to hold the final chunk examples
        current_chunk = []  # list of sentences for the current chunk
        for sentence in sentences:
            # Candidate chunk by adding the current sentence
            candidate = " ".join(current_chunk + [sentence]).strip()
            # Tokenize the candidate text without truncation
            tokenized_candidate = tokenizer(candidate)["input_ids"]
            if len(tokenized_candidate) <= args.max_seq_length:
                # If the candidate is within the limit, update current chunk.
                current_chunk.append(sentence)
            else:
                # If adding this sentence would exceed the max, finalize the current chunk if non-empty.
                if current_chunk:
                    chunk_text_str = " ".join(current_chunk).strip()
                    # Tokenize the finalized chunk with truncation (for safety)
                    tokenized_chunk = tokenizer(chunk_text_str, truncation=True, max_length=args.max_seq_length, padding='max_length', return_special_tokens_mask=True)

                    chunks.append({
                        "input_ids": tokenized_chunk['input_ids'],
                        "attention_mask": tokenized_chunk['attention_mask']
                    })
                # Start a new chunk with the current sentence.
                # Optionally, check if the sentence itself exceeds the limit:
                tokenized_sentence = tokenizer(sentence, truncation=False)["input_ids"]
                current_chunk = [sentence] if len(tokenized_sentence) <= args.max_seq_length else []
                # If the sentence on its own exceeds max_seq_length, you might want to
                # either split it further (not covered here) or skip it.

        # After processing all sentences, add any remaining text as a chunk.
        if current_chunk:
            chunk_text_str = " ".join(current_chunk).strip()
            tokenized_output = tokenizer(chunk_text_str, truncation=True, max_length=args.max_seq_length, padding='max_length', return_special_tokens_mask=True)

            tokenized_chunk = tokenized_output["input_ids"]
            attention_mask = tokenized_output["attention_mask"]

            chunks.append({
                "input_ids": tokenized_chunk,
                "attention_mask": attention_mask
            })
        return chunks
    except:
        tokenized_output=tokenizer(full_text, truncation=True,
           padding='max_length', max_length=args.max_seq_length)

        return [{
                "input_ids": tokenized_output["input_ids"],
                "attention_mask": tokenized_output['attention_mask']
        }]

# Assume raw_datasets has been loaded as before.
print("Chunking training dataset manually...")
all_train_chunks = []
for example in tqdm.tqdm(raw_datasets["train"]):
    chunks = chunk_text(example)  # This returns a list of dicts.
    all_train_chunks.extend(chunks)
chunked_train = Dataset.from_list(all_train_chunks)

print("Chunking validation dataset manually...")
all_validation_chunks = []
for example in tqdm.tqdm(raw_datasets["validation"]):
    chunks = chunk_text(example)
    all_validation_chunks.extend(chunks)
chunked_validation = Dataset.from_list(all_validation_chunks)

# Combine the new splits into a single dataset dictionary.
chunked_dataset = DatasetDict({"train": chunked_train, "validation": chunked_validation})

# Save the chunked dataset to disk.
print(f"Saving chunked dataset to disk at {args.save_dir}...")
chunked_dataset.save_to_disk(args.save_dir)
print("Dataset saved. You can now load it later using `load_from_disk` in your training script.")
