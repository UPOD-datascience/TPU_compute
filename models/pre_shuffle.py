import argparse
from datasets import load_dataset, DatasetDict
import os
import sys
import shutil
import gc

try:
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
except:
    print("No TPU_PROCESS_ADDRESSES or CLOUD_TPU_TASK_ID to remove")

def shuffle_and_save_dataset(dataset, output_path, seed=42):
    # Shuffle only the training dataset
    shuffled_train = dataset['train'].shuffle(seed=seed)

    # Save the shuffled training dataset and the original validation dataset
    shuffled_dataset = DatasetDict({
        'train': shuffled_train,
        'validation': dataset['validation']
    })
    print("Saving shuffled data to disk", flush=True)
    shuffled_dataset.save_to_disk(output_path)
    print(f"Shuffled dataset saved to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_format", default="json", choices=["json", "parquet"])
    parser.add_argument("--shuffle_dataset_path", type=str, default="/home/bob/tmp/shuffle.parquet")
    args = parser.parse_args()

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

    # Update the dataset directory to the shuffled dataset path
    # Clear the dataset from memory
    del dataset
    gc.collect()
    print("Cleared shuffled dataset from master's memory")
