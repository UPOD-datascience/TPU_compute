#!/usr/bin/env python
from posixpath import extsep
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
import argparse
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import pysbd
import tqdm
import json
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

# Create an argument parser to accept file paths, tokenizer model, and max_seq_length.
argparser = argparse.ArgumentParser(description="Preprocess and chunk dataset documents.")
argparser.add_argument("--data_bucket", type=str, required=True, help="Google Cloud Storage bucket where the training and validation JSON files are stored.")
argparser.add_argument("--train_loc", type=str, required=True, help="Path to the training JSON file.")
argparser.add_argument("--validation_loc", type=str, required=True, help="Path to the validation JSON file.")
argparser.add_argument("--save_dir_local", type=str, required=True, help="Directory where the preprocessed dataset will be saved.")
argparser.add_argument("--save_dir_gcs", type=str, required=False, help="Google Cloud Storage path where the preprocessed dataset will be saved.")
argparser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Pretrained tokenizer model name or path (e.g., 'microsoft/deberta-v3-base').")
argparser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length for each chunk.")
argparser.add_argument("--write_mode", type=str, required=True, choices=["jsonl", "parquet"], help="Format to save the preprocessed dataset.")
argparser.add_argument("--debug_mode", action="store_true", help="Run the script in debug mode.")
argparser.add_argument('--autotokenizer', type=bool, default=False, help='Automatically download and cache the tokenizer from the Hugging Face model hub.')

args = argparser.parse_args()

print(f"Checking if save_dir_local exists and otherwise make...")
os.makedirs(os.path.join(args.save_dir_local, 'train'), exist_ok=True)
os.makedirs(os.path.join(args.save_dir_local, 'validation'), exist_ok=True)

train_loc_dir = os.path.join(args.save_dir_local, 'train')
val_loc_dir = os.path.join(args.save_dir_local, 'validation')

# get lists of files present
current_train_files = os.listdir(train_loc_dir)
local_train_files = [os.path.join(train_loc_dir, f) for f in current_train_files]
current_validation_files = os.listdir(val_loc_dir)
local_validation_files = [os.path.join(val_loc_dir, f) for f in current_validation_files]

if args.autotokenizer:
    print(f"Loading AutoTokenizer from {args.tokenizer_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True, truncation=True, model_max_length=args.max_seq_length)
else:
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

features = Features({
    "id": Value("string"),
    "text": Value("string"),
    "source": Value("string"),
    "approx_token_counts_original": Value("int64"),
    "approx_token_counts_translated": Value("int64"),
})

print("Connecting to Google Cloud Storage...")
client = storage.Client.from_service_account_json('../gsa.json')
bucket = client.get_bucket(args.data_bucket.split('gs://')[-1])

local_train_files = []
counter = 0
print(f"Downloading training data from {args.train_loc}...")
for blob in bucket.list_blobs(prefix=args.train_loc.split('gs://')[-1].split("/")[1]):
    local_file_path = os.path.join(train_loc_dir, blob.name.split('/')[-1])
    current_files = os.listdir(train_loc_dir)
    local_train_files = [os.path.join(train_loc_dir, f) for f in current_files]
    if ('.json' in local_file_path) and (local_file_path.split('/')[-1] not in current_files):
        print(f'Downloading to {local_file_path}')
        blob.download_to_filename(local_file_path)
        local_train_files.append(local_file_path)
        counter += 1
        if (counter == 1) & (args.debug_mode):
            break

local_validation_files = []
counter = 0
print(f"Downloading validation data from {args.validation_loc}...")
for blob in bucket.list_blobs(prefix=args.validation_loc.split('gs://')[-1].split("/")[1]):
    local_file_path = os.path.join(val_loc_dir, blob.name.split('/')[-1])
    current_files = os.listdir(val_loc_dir)
    local_validation_files = [os.path.join(val_loc_dir, f) for f in current_files]
    if ('.json' in local_file_path) and (local_file_path.split('/')[-1] not in current_files):
        print(f'Downloading to {local_file_path}')
        blob.download_to_filename(local_file_path)
        local_validation_files.append(local_file_path)
        counter += 1
        if (counter == 1) & (args.debug_mode):
            break

data_files = {
                "train": local_train_files,
                "validation": local_validation_files
}
print("Data files loaded.")
print(data_files)
print("Loading dataset...")

# TODO:
# Split in two lists of datasets
# train and validation
# per file in the list we create a dataset
# raw_datasets = {'train': [dataset1, dataset2, ...], 'validation': [dataset1, dataset2, ...]}
#
def create_dataset_from_file(filename):
    return Dataset.from_file(filename, features=features,
        keep_in_memory=False, streaming=True, num_proc=None)

raw_datasets = {
    'train': [(create_dataset_from_file(file), os.path.splitext(os.path.basename(file))[0]) for file in data_files['train']],
    'validation': [(create_dataset_from_file(file), os.path.splitext(os.path.basename(file))[0]) for file in data_files['validation']]
}

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
def chunked_examples_generator(raw_datasets, split_name):
    assert(split_name is not None)
    for example in tqdm.tqdm(raw_datasets[split_name]):
        chunks = chunk_text(example)  # This returns a list of dicts.
        for chunk in chunks:
            yield chunk

def write_chunks_to_disk(generator, output_file):
    with open(output_file, 'w') as f:
        for chunk in generator:
            f.write(json.dumps(chunk) + '\n')

def write_chunks_to_parquet(generator, output_file):
    """Writes data from a generator to a Parquet file in chunks."""
    schema = pa.schema([
        pa.field("input_ids", pa.int64()),
        pa.field("attention_mask", pa.int64())
    ])

    with pq.ParquetWriter(output_file, compression='snappy', schema=schema) as writer:  # Choose compression
        for batch in generator: # Assuming your generator yields batches of data
            table = pa.Table.from_pydict(batch, schema=schema) # Create a pyarrow table from batch
            writer.write_table(table)


###################################################################
################### Chunking the training dataset #################

upload_bucket = client.get_bucket(args.save_dir_gcs.replace("gs://",""))
# get list of files in the bucket under the dataset folder
existing_files = [blob.name for blob in upload_bucket.list_blobs(prefix="dataset/")]

print("Chunking training dataset manually...")
train_files = []
if args.write_mode == 'parquet':
    # TODO: loop over all datasets in raw_datasets['train'], check that the output file does not exist
    for dataset, dataset_name in raw_datasets['train']:
        if not any([dataset in ef for ef in existing_files]):
            output_file_train = os.path.join(args.save_dir_local, f'{dataset_name}_chunked_train_{args.max_seq_length}.parquet')
            if not os.path.exists(output_file_train):
                write_chunks_to_parquet(chunked_examples_generator(dataset, "train"), output_file_train)
                train_files.append((output_file_train,dataset_name))
            else:
                print(f"{output_file_train} already exists locally, skipping chunking.")
        else:
            print(f"{dataset_name} already exists in the GCS bucket, skipping chunking.")

elif args.write_mode == 'jsonl':
    for dataset, dataset_name in raw_datasets['train']:
        if not any([dataset in ef for ef in existing_files]):
            output_file_train = os.path.join(args.save_dir_local, f'{dataset_name}_chunked_train_{args.max_seq_length}.jsonl')
            if not os.path.exists(output_file_train):
                write_chunks_to_disk(chunked_examples_generator(dataset, "train"), output_file_train)
                train_files.append((output_file_train,dataset_name))
            else:
                print(f"{output_file_train} already exists, skipping chunking.")
        else:
            print(f"{dataset_name} already exists in the GCS bucket, skipping chunking.")
else:
    raise ValueError(f"Invalid write mode: {args.write_mode}")

print("Chunking validation dataset manually...")
validation_files = []
if args.write_mode == 'parquet':
    for dataset, dataset_name in raw_datasets['validation']:
        if not any([dataset in ef for ef in existing_files]):
            output_file_validation = os.path.join(args.save_dir_local, f'{dataset_name}_chunked_validation_{args.max_seq_length}.parquet')
            if not os.path.exists(output_file_validation):
                write_chunks_to_parquet(chunked_examples_generator(dataset, "validation"), output_file_validation)
                validation_files.append((output_file_validation,dataset_name))
            else:
                print(f"{output_file_validation} already exists, skipping chunking.")
        else:
            print(f"{dataset_name} already exists in the GCS bucket, skipping chunking.")
elif args.write_mode == 'jsonl':
    for dataset, dataset_name in raw_datasets['validation']:
        if not any([dataset in ef for ef in existing_files]):
            output_file_validation = os.path.join(args.save_dir_local, f'{dataset_name}_chunked_validation_{args.max_seq_length}.jsonl')
            if not os.path.exists(output_file_validation):
                write_chunks_to_disk(chunked_examples_generator(dataset, "validation"), output_file_validation)
                validation_files.append((output_file_validation,dataset_name))
            else:
                print(f"{output_file_validation} already exists, skipping chunking.")
        else:
            print(f"{dataset_name} already exists in the GCS bucket, skipping chunking.")
###############

ext = "json" if args.write_mode == 'jsonl' else "parquet"

upload_bucket = client.get_bucket(args.save_dir_gcs.replace("gs://",""))
for output_file_train, dataset_name in train_files:
    blob = upload_bucket.blob(f"dataset/{dataset_name}_train_{args.max_seq_length}.{ext}")
    print(f"Upload chunked training dataset to disk to {args.save_dir_gcs.replace("gs://","")}{"/dataset"}...")
    blob.upload_from_filename(output_file_train)

for output_file_validation, dataset_name in validation_files:
    blob = upload_bucket.blob(f"dataset/{dataset_name}_validation_{args.max_seq_length}.{ext}")
    print(f"Upload chunked training dataset to disk to {args.save_dir_gcs.replace("gs://","")}{"/dataset"}...")
    blob.upload_from_filename(output_file_validation)
    # remove local file

print("Removing local files..")
for f in local_train_files + local_validation_files:
    try:
        os.remove(f)
    except Exception as e:
        print(f"Error removing {f}: {e}")
