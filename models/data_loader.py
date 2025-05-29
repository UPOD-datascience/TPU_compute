import os
import json
import random
import logging
import io # For GCS blob streaming

import torch
from torch.utils.data import IterableDataset, get_worker_info

# Configure logging - If already configured at the application entry point, this might be redundant
# or could be adjusted (e.g., getLogger instead of basicConfig).
# For a library file, it's often better to let the application configure logging.
# However, the original file had this, so keeping it for consistency with the project.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global placeholder for GCS storage module, loaded on demand
_gcs_storage_module = None
_gcs_module_checked = False

def _try_load_gcs_storage_module():
    """Attempts to load the google.cloud.storage module and caches it."""
    global _gcs_storage_module, _gcs_module_checked
    if _gcs_module_checked:
        return _gcs_storage_module

    try:
        from google.cloud import storage
        _gcs_storage_module = storage
        logging.debug("Successfully imported google-cloud-storage.")
    except ImportError:
        _gcs_storage_module = None
        logging.debug("google-cloud-storage not found.")
    _gcs_module_checked = True
    return _gcs_storage_module


class DirectoryShardedShuffleDataset(IterableDataset):
    """
    An IterableDataset that reads .jsonl files from a local directory or GCS path,
    optionally shuffles the file order, and optionally shuffles lines from the files
    using a running buffer.
    """
    def __init__(self, data_path: str, buffer_size: int = 25_000,
                 file_extension: str = ".json", seed: int = 42,
                 gcs_project: str|None = None, shuffle_files: bool = True,
                 shuffle_lines: bool = True):
        """
        Args:
            data_path (str): Path to the directory containing .jsonl files (local or gs://bucket/prefix).
            buffer_size (int, optional): Size of the buffer used for shuffling lines. Defaults to 25_000.
                                         If 0, no line shuffling or buffering occurs.
            file_extension (str, optional): Extension of the data files. Defaults to ".jsonl".
            seed (int, optional): Random seed for shuffling. Defaults to 42.
            gcs_project (str, optional): Google Cloud Project ID, used if data_path is GCS and project isn't auto-discoverable.
            shuffle_files (bool, optional): Whether to shuffle the order of files. Defaults to True.
            shuffle_lines (bool, optional): Whether to shuffle lines within the buffer. Defaults to True.
                                           Only effective if buffer_size > 0.
        """
        super().__init__()
        self.data_path = data_path
        self.buffer_size = buffer_size
        self.file_extension = file_extension
        self.seed = seed
        self.gcs_project = gcs_project
        self.shuffle_files = shuffle_files
        self.shuffle_lines = shuffle_lines

        self.is_gcs_path = self.data_path.startswith("gs://")
        self.gcs_client = None
        self.gcs_bucket_name = None
        self.gcs_prefix = ""

        if self.is_gcs_path:
            storage_module = _try_load_gcs_storage_module()
            if storage_module is None:
                raise ImportError(
                    "A GCS path was provided, but the 'google-cloud-storage' library "
                    "is not installed or couldn't be imported. Please install it to use GCS features."
                )
            try:
                self.gcs_client = storage_module.Client(project=self.gcs_project)
            except Exception as e:
                logging.error(f"Failed to initialize GCS client with project '{self.gcs_project}': {e}")
                raise

            parts = self.data_path[5:].split("/", 1)
            self.gcs_bucket_name = parts[0]
            if len(parts) > 1 and parts[1]: # Ensure prefix is not empty if present
                self.gcs_prefix = parts[1]
                # For directory listing, prefix should ideally end with /
                # This class expects a "directory" of files.
                if not self.gcs_prefix.endswith("/") and self.gcs_prefix:
                    self.gcs_prefix += "/"
            logging.info(f"Using GCS path: bucket='{self.gcs_bucket_name}', prefix='{self.gcs_prefix}'")
        else: # Local path
            if not os.path.isdir(self.data_path):
                raise ValueError(f"Local directory not found: {self.data_path}")
            logging.info(f"Using local path: {self.data_path}")

        self.file_paths = self._find_files()
        if not self.file_paths:
            logging.warning(f"No files with extension '{self.file_extension}' found in '{self.data_path}'.")
        else:
            logging.info(f"Found {len(self.file_paths)} files in '{self.data_path}' with extension '{self.file_extension}'.")

    def _find_files(self):
        paths = []
        if self.is_gcs_path:
            if not self.gcs_client or not self.gcs_bucket_name:
                logging.error("GCS client or bucket name not initialized for _find_files.")
                return []
            try:
                # bucket_obj = self.gcs_client.bucket(self.gcs_bucket_name) # Not strictly needed for list_blobs by name
                blobs = self.gcs_client.list_blobs(self.gcs_bucket_name, prefix=self.gcs_prefix)
                for blob in blobs:
                    # Ensure it's a file (not a "folder" itself) and has the right extension
                    if blob.name.endswith(self.file_extension) and not blob.name.endswith('/'):
                        paths.append(f"gs://{self.gcs_bucket_name}/{blob.name}")
            except Exception as e:
                logging.error(f"Error listing GCS blobs in gs://{self.gcs_bucket_name}/{self.gcs_prefix}: {e}")
        else: # Local file system
            for root, _, files in os.walk(self.data_path):
                for file_name in files:
                    if file_name.endswith(self.file_extension):
                        paths.append(os.path.join(root, file_name))
        return paths

    def _read_lines_from_file(self, file_path, worker_id_log):
        if self.is_gcs_path:
            if not self.gcs_client:
                logging.error(f"Worker {worker_id_log}: GCS client not available for reading {file_path}.")
                return
            try:
                path_parts = file_path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1]

                bucket = self.gcs_client.bucket(bucket_name)
                blob_obj = bucket.blob(blob_name)

                with blob_obj.open("rt", encoding="utf-8") as f: # Requires io module
                    for line in f:
                        yield line.rstrip("\n\r") # Remove newlines as f often keeps them
            except Exception as e:
                logging.error(f"Worker {worker_id_log}: Error reading GCS file {file_path}: {e}. Skipping.")
                return
        else: # Local file system
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line.rstrip("\n\r") # Remove newlines
            except FileNotFoundError:
                logging.error(f"Worker {worker_id_log}: File not found: {file_path}. Skipping.")
            except Exception as e:
                logging.error(f"Worker {worker_id_log}: Error reading local file {file_path}: {e}. Skipping.")

    def __iter__(self):
        worker_info = get_worker_info()
        current_seed = self.seed
        worker_id_log = "0 (main)"
        if worker_info is not None:
            current_seed += worker_info.id
            worker_id_log = str(worker_info.id)

        rng = random.Random(current_seed)

        if not self.file_paths:
            return iter([])

        epoch_file_paths = list(self.file_paths)
        if self.shuffle_files:
            rng.shuffle(epoch_file_paths)

        logging.info(f"Worker {worker_id_log}: Processing {len(epoch_file_paths)} files (shuffled: {self.shuffle_files}) with seed {current_seed}.")

        buffer = []
        for file_path in epoch_file_paths:
            logging.debug(f"Worker {worker_id_log}: Loading file {file_path}")
            line_number = 0
            for line in self._read_lines_from_file(file_path, worker_id_log):
                line_number += 1
                try:
                    item = json.loads(line) # line should not have trailing newline here
                    if self.buffer_size > 0:
                        if len(buffer) < self.buffer_size:
                            buffer.append(item)
                        else: # Buffer is full
                            if self.shuffle_lines:
                                idx_to_yield = rng.randint(0, self.buffer_size - 1)
                                yield buffer[idx_to_yield]
                                buffer[idx_to_yield] = item
                            else: # FIFO if not shuffling lines
                                yield buffer.pop(0)
                                buffer.append(item)
                    else: # No buffer (buffer_size is 0 or less)
                        yield item
                except json.JSONDecodeError as e:
                    logging.warning(f"Worker {worker_id_log}: Skipping line {line_number} due to JSONDecodeError in {file_path}: {e} - Line: '{line.strip()}'")
                except Exception as e:
                    logging.error(f"Worker {worker_id_log}: Unexpected error processing line {line_number} in {file_path}: {e} - Line: '{line.strip()}'")

        # Yield remaining items in the buffer
        if self.buffer_size > 0 and buffer:
            if self.shuffle_lines:
                rng.shuffle(buffer)
            logging.debug(f"Worker {worker_id_log}: Yielding {len(buffer)} remaining items from buffer (shuffled: {self.shuffle_lines}).")
            while buffer:
                yield buffer.pop(0)


class ShardedShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, num_shards, shard_id, shuffle_buffer_size=25_000, max_steps=None, batch_size=1):
        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_steps = max_steps
        self.batch_size = batch_size

    def __iter__(self):
        buffer = []
        items_yielded = 0
        # Calculate max_items carefully if batch_size might be 0 or max_steps is None
        if self.max_steps is not None and self.batch_size > 0:
            max_items = self.max_steps * self.batch_size
        else:
            max_items = float('inf')


        for i, item in enumerate(self.dataset):
            if items_yielded >= max_items:
                break
            if i % self.num_shards == self.shard_id:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer) # Uses global random, consider passing/creating an RNG
                    while buffer and items_yielded < max_items:
                        items_yielded += 1
                        yield buffer.pop(0)

        # Yield remaining items
        if buffer: # Check if buffer is not empty
            random.shuffle(buffer) # Uses global random
            while buffer and items_yielded < max_items:
                items_yielded += 1
                yield buffer.pop(0)

def tokenize_and_group(example, tokenizer, max_seq_length):
    """
    Helper function to tokenize and group a single example.
    Applied to items from DirectoryShardedShuffleDataset.

    Args:
        example: A dictionary with 'text' key (from .jsonl file)
        tokenizer: The tokenizer to use
        max_seq_length: Maximum sequence length for grouping

    Returns:
        A dictionary with input_ids, attention_mask, and labels
    """
    # Extract text from the example
    if 'text' not in example:
        # Find the first string field if 'text' doesn't exist
        for key, value in example.items():
            if isinstance(value, str):
                text = value
                break
        else:
            # If no string field found, use a default empty string
            text = ""
            logging.warning(f"No text field found in example: {example}")
    else:
        text = example['text']

    # Tokenize the text
    tokenized = tokenizer(text, truncation=True, max_length=max_seq_length)

    # Group into full length sequences if needed
    # If the sequence is already full-length or group_texts functionality is elsewhere,
    # you might not need this part

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].copy()  # For masked language modeling
    }

def iterate_batches(data_iter, batch_size):
    batch = []
    for item in data_iter:
        batch.append(item)
        if len(batch) == batch_size:
            # Convert batch to tensors and yield
            yield {
                k: torch.tensor([d[k] for d in batch])
                for k in batch[0].keys()
            }
            batch = []
    # Don't forget the last partial batch
    if batch:
        yield {
            k: torch.tensor([d[k] for d in batch])
            for k in batch[0].keys()
        }
