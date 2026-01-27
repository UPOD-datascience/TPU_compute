#!/usr/bin/env python
# Incoming: location with .json's, containing 'text', 'id', 'source', 'approx_token_counts_original', 'approx_token_counts_translated'

# Outgoing: combined .parquet's or jsons with source is existing source or filename of dataset
import argparse
import json
import os
from posixpath import extsep

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysbd
import tqdm
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from google.cloud import storage
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Create an argument parser to accept file paths, tokenizer model, and max_seq_length.
argparser = argparse.ArgumentParser(
    description="Preprocess and chunk dataset documents."
)
argparser.add_argument(
    "--data_bucket",
    type=str,
    required=True,
    help="Google Cloud Storage bucket where the training and validation JSON files are stored.",
)
argparser.add_argument(
    "--train_loc", type=str, required=True, help="Path to the training JSON file."
)
argparser.add_argument(
    "--validation_loc",
    type=str,
    required=True,
    help="Path to the validation JSON file.",
)
argparser.add_argument(
    "--save_dir_local",
    type=str,
    required=True,
    help="Directory where the preprocessed dataset will be saved.",
)
argparser.add_argument(
    "--save_dir_gcs",
    type=str,
    required=False,
    help="Google Cloud Storage path where the preprocessed dataset will be saved.",
)
argparser.add_argument(
    "--write_mode",
    type=str,
    required=True,
    choices=["jsonl", "parquet"],
    help="Format to save the preprocessed dataset.",
)
argparser.add_argument(
    "--debug_mode", action="store_true", help="Run the script in debug mode."
)
argparser.add_argument(
    "--validation_only",
    action="store_true",
    help="Only preprocess the validation data.",
)
argparser.add_argument(
    "--in_memory", action="store_true", help="Load all file in memory"
)
args = argparser.parse_args()

print(f"Checking if save_dir_local exists and otherwise make...")
os.makedirs(os.path.join(args.save_dir_local, "train"), exist_ok=True)
os.makedirs(os.path.join(args.save_dir_local, "validation"), exist_ok=True)

train_loc_dir = os.path.join(args.save_dir_local, "train")
val_loc_dir = os.path.join(args.save_dir_local, "validation")

# get lists of files present
current_train_files = os.listdir(train_loc_dir)
local_train_files = [os.path.join(train_loc_dir, f) for f in current_train_files]
current_validation_files = os.listdir(val_loc_dir)
local_validation_files = [
    os.path.join(val_loc_dir, f) for f in current_validation_files
]


# Define data_files dictionary for the dataset loader.

features = Features(
    {
        "id": Value("string"),
        "text": Value("string"),
        "source": Value("string"),
        "approx_token_counts_original": Value("int64"),
        "approx_token_counts_translated": Value("int64"),
    }
)

pa_schema = pa.schema(
    [
        pa.field(name, pa.string())
        if feature.dtype == "string"
        else pa.field(name, pa.int64(), nullable=True)
        for name, feature in features.items()
    ]
)

# Cache for storage client
_storage_client = None


def get_storage_client() -> storage.Client:
    """
    Get a Google Cloud Storage client using Application Default Credentials.

    This works with:
    - `gcloud auth application-default login` (for local development)
    - Service account when running on GCP (automatic)
    - GOOGLE_APPLICATION_CREDENTIALS environment variable (if set)

    The client is cached for reuse across multiple calls.
    """
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


print("Connecting to Google Cloud Storage...")
client = get_storage_client()
bucket = client.get_bucket(args.data_bucket.split("gs://")[-1])

local_train_files = []
if args.validation_only == False:
    counter = 0
    print(f"Downloading training data from {args.train_loc}...")
    for blob in bucket.list_blobs(
        prefix=args.train_loc.split("gs://")[-1].split("/")[1]
    ):
        local_file_path = os.path.join(train_loc_dir, blob.name.split("/")[-1])
        current_files = os.listdir(train_loc_dir)
        local_train_files = [os.path.join(train_loc_dir, f) for f in current_files]
        if (".json" in local_file_path) and (
            local_file_path.split("/")[-1] not in current_files
        ):
            print(f"Downloading to {local_file_path}")
            blob.download_to_filename(local_file_path)
            local_train_files.append(local_file_path)
            counter += 1
            if (counter == 1) & (args.debug_mode):
                break

local_validation_files = []
counter = 0
print(f"Downloading validation data from {args.validation_loc}...")
for blob in bucket.list_blobs(
    prefix=args.validation_loc.split("gs://")[-1].split("/")[1]
):
    local_file_path = os.path.join(val_loc_dir, blob.name.split("/")[-1])
    current_files = os.listdir(val_loc_dir)
    local_validation_files = [os.path.join(val_loc_dir, f) for f in current_files]
    if (".json" in local_file_path) and (
        local_file_path.split("/")[-1] not in current_files
    ):
        print(f"Downloading to {local_file_path}")
        blob.download_to_filename(local_file_path)
        local_validation_files.append(local_file_path)
        counter += 1
        if (counter == 1) & (args.debug_mode):
            break

data_files = {"train": local_train_files, "validation": local_validation_files}
print("Data files loaded.")
print(data_files)
print("Loading dataset...")


# TODO:
# Split in two lists of datasets
# train and validation
# per file in the list we create a dataset
# raw_datasets = {'train': [dataset1, dataset2, ...], 'validation': [dataset1, dataset2, ...]}
#
def create_dataset_from_file(filename, split="train"):
    ds = load_dataset(
        "json",
        data_files={split: filename},
        features=features,
        keep_in_memory=args.in_memory,
        streaming=(not args.in_memory),
        num_proc=None,
    )

    def update_source(example):
        if example["source"] == "not available":
            example["source"] = os.path.splitext(os.path.basename(filename))[0]
        return example

    ds = ds.map(update_source)
    return ds[split]


raw_datasets = {
    "train": [
        (
            create_dataset_from_file(file, "train"),
            os.path.splitext(os.path.basename(file))[0],
        )
        for file in data_files["train"]
    ],
    "validation": [
        (
            create_dataset_from_file(file, "validation"),
            os.path.splitext(os.path.basename(file))[0],
        )
        for file in data_files["validation"]
    ],
}

if args.in_memory:
    combined_datasets = {}
    for split in ["train", "validation"]:
        if raw_datasets[split]:  # Check if the list is not empty
            first_dataset = raw_datasets[split][0][0]
            if (
                first_dataset is not None
            ):  # Additional check to ensure the dataset is not None
                combined_datasets[split] = Dataset.from_dict(
                    {
                        key: [
                            item
                            for dataset, _ in raw_datasets[split]
                            if dataset is not None
                            for item in dataset[key]
                        ]
                        for key in first_dataset.features
                    }
                )
            else:
                print(f"Warning: No valid datasets found for {split} split.")
        else:
            print(f"Warning: No datasets found for {split} split.")

    # Write to parquet or jsonl
    for split, dataset in combined_datasets.items():
        if args.write_mode == "parquet":
            output_file = os.path.join(args.save_dir_local, f"{split}.parquet")
            dataset.to_parquet(output_file)
        elif args.write_mode == "jsonl":
            output_file = os.path.join(args.save_dir_local, f"{split}.jsonl")
            dataset.to_json(output_file, lines=True)
else:
    # Process and write datasets
    for split in ["train", "validation"]:
        if raw_datasets[split]:
            if args.write_mode == "parquet":
                output_file = os.path.join(args.save_dir_local, f"{split}.parquet")
                writer = None
                batch = []
                with pq.ParquetWriter(
                    output_file, compression="snappy", schema=pa_schema
                ) as writer:
                    for dataset, name in tqdm.tqdm(raw_datasets[split]):
                        for item in dataset:
                            batch.append(item)
                            if len(batch) == 5_000:
                                table = pa.Table.from_pandas(
                                    pd.DataFrame(batch), schema=pa_schema
                                )
                                writer.write_table(table, row_group_size=5_000)
                                batch = []
                        if batch:  # Write any remaining items
                            table = pa.Table.from_pandas(
                                pd.DataFrame(batch), schema=pa_schema
                            )
                            writer.write_table(table, row_group_size=5_000)

            elif args.write_mode == "jsonl":
                output_file = os.path.join(args.save_dir_local, f"{split}.jsonl")
                with open(output_file, "w") as f:
                    for dataset, name in tqdm.tqdm(
                        raw_datasets[split], desc=f"Processing {split} datasets"
                    ):
                        for item in dataset:
                            # Convert the item to a dictionary
                            item_dict = dict(item)
                            # Ensure integers are JSON serializable
                            for key in [
                                "approx_token_counts_original",
                                "approx_token_counts_translated",
                            ]:
                                try:
                                    if key in item_dict:
                                        item_dict[key] = int(item_dict[key])
                                except:
                                    if key in item_dict:
                                        item_dict[key] = -1

                            f.write(json.dumps(item_dict) + "\n")
        else:
            print(f"Warning: No datasets found for {split} split.")
