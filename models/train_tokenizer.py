# train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer,BertWordPieceTokenizer
from transformers import DebertaV2TokenizerFast
import os
import argparse
import json
import re
import io
from google.cloud import storage
import pandas as pd
import ftfy
import pyarrow.parquet as pq

from typing import List, Iterator

def clean_text(text):
    '''
    - remove spurious repetitions of characters, punctuation whitespace and linebreaks
    - remove spurious repetitions of words
    '''
    # TODO: Add more/improve cleaning steps as needed
    re_spurious_chars = re.compile(r'([^\w])\1{3,}')
    re_spurious_words = re.compile(r'(\b\w+\b)\1{4,}')
    re_multispace = re.compile(r'\s{2,}')

    # replace spurious repetitions of characters, punctuation, whitespace and linebreaks with a single instance
    text = re_spurious_chars.sub(r'\1', text)
    text = re_spurious_words.sub(r'\1', text)
    text = re_multispace.sub(' ', text)
    text = ftfy.fix_encoding(text)

    return text

# Function to read JSONL file and extract text data
def read_jsonl(file_path):
    texts = []
    if file_path.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = file_path.split("/")[2]
        blob_name = "/".join(file_path.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text(encoding='utf-8')
        for line in content.splitlines():
            data = json.loads(line)
            if data is not None:
                texts.append(data['text'])  # Adjust the key based on your JSON structure
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data is not None:
                    texts.append(data['text'])  # Adjust the key based on your JSON structure
    return texts

def read_jsonl_iterator(file_path, chunk_size=1024*1024):
    """
    Iterator that reads a text file line by line and yields each line as a string.

    Args:
        file_path (str): Path to the text file

    Yields:
        str: Each line from the text file
    """
    if file_path.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = file_path.split("/")[2]
        blob_name = "/".join(file_path.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Create in-memory file-like object
        file_obj = io.BytesIO()
        # Download to the file object
        blob.download_to_file(file_obj)
        # Reset file pointer to beginning
        file_obj.seek(0)

        # Convert to TextIOWrapper for line-by-line reading
        text_file = io.TextIOWrapper(file_obj, encoding='utf-8', newline='\n')

        # Read line by line
        for line in text_file:
            d = json.loads(line)
            if d is not None:
                yield d['text']

    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                if d is not None:
                    yield d['text']

# Function to read TXT file and extract text data
def read_txt(file_path):
    texts = []
    if file_path.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = file_path.split("/")[2]
        blob_name = "/".join(file_path.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text(encoding='utf-8')
        texts = content.splitlines()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    return texts

def read_txt_iterator(file_path, chunk_size=1024*1024):
    """
    Iterator that reads a text file line by line and yields each line as a string.

    Args:
        file_path (str): Path to the text file

    Yields:
        str: Each line from the text file
    """
    if file_path.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = file_path.split("/")[2]
        blob_name = "/".join(file_path.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Create in-memory file-like object
        file_obj = io.BytesIO()
        # Download to the file object
        blob.download_to_file(file_obj)
        # Reset file pointer to beginning
        file_obj.seek(0)

        # Convert to TextIOWrapper for line-by-line reading
        text_file = io.TextIOWrapper(file_obj, encoding='utf-8', newline='\n')

        # Read line by line
        for line in text_file:
            yield line

    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

def read_parquet(file_path):
    texts = []
    if file_path.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = file_path.split("/")[2]
        blob_name = "/".join(file_path.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(content))
        texts = df['text'].tolist()
    else:
        df = pd.read_parquet(file_path)
        texts = df['text'].tolist()
    return texts

def read_parquet_iterator(file_path, batch_size=10_000):
    """
    Iterator that reads a Parquet file in batches and yields the 'text' field from each row.

    Args:
        file_path (str): Path to the Parquet file
        batch_size (int): Number of rows to read in each batch

    Yields:
        str: The 'text' field from each row
    """
    # Open the Parquet file
    parquet_file = pq.ParquetFile(file_path)

    # Iterate through batches
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert batch to pandas DataFrame
        batch_df = batch.to_pandas()

        # Yield the 'text' field from each row in the batch
        for text in batch_df['text']:
            yield text

def combine_iterators(iterators: List[Iterator[str]])->Iterator[str]:
    """Combine multiple iterators into a single iterator."""
    for iterator in iterators:
        for item in iterator:
            yield item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='~/tokenizer', required=True)
    parser.add_argument("--vocab_size", type=int, default=50_368)
    parser.add_argument("--min_frequency", type=int, default=100)
    parser.add_argument("--iterative", action='store_true')
    parser.add_argument("--tokenizer_type", type=str, choices=['bbpe', 'sentencebpe', 'debertav2', 'bertwordpiece'], default='bbpe')
    args = parser.parse_args()

    print("Getting file list..")
    if args.data_dir.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = args.data_dir.split("/")[2]
        prefix = "/".join(args.data_dir.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        corpus_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(('.jsonl', '.txt', '.parquet'))]
    else:
        corpus_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(('.jsonl', '.txt', '.parquet'))]

    print("Reading files..")
    # Combine texts from all corpus files
    if not args.iterative:
        all_texts = []
        for file_path in corpus_files:
            if file_path.endswith('.jsonl'):
                all_texts.extend(read_jsonl(file_path))
            elif file_path.endswith('.txt'):
                all_texts.extend(read_txt(file_path))
            elif file_path.endswith('.parquet'):
                all_texts.extend(read_parquet(file_path))
        # Clean the texts
        print("Cleaning texts..")
        all_texts = [clean_text(text) for text in all_texts]
    else:
        text_iterators = []
        for file_path in corpus_files:
            if file_path.endswith('.jsonl'):
                text_iterators.append(read_jsonl_iterator(file_path))
            elif file_path.endswith('.txt'):
                text_iterators.append(read_txt_iterator(file_path))
            elif file_path.endswith('.parquet'):
                text_iterators.append(read_parquet_iterator(file_path))

        all_texts_iterator = combine_iterators(text_iterators)
        all_texts = map(clean_text, all_texts_iterator)

    if args.tokenizer_type == 'bbpe':
        # Initialize a ByteLevel BPE tokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer._tokenizer.post_processor.verbose = True
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "<s>", "</s>", "<pad>", "<unk>", "<mask>"
        ])
    elif args.tokenizer_type == 'sentencebpe':
        # Initialize a SentencePiece BPE tokenizer
        tokenizer = SentencePieceBPETokenizer()
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "<s>", "</s>", "<pad>", "<unk>", "<mask>"
        ])
    elif args.tokenizer_type in  ['bertwordpiece', 'debertav2']:
        # Initialize a DeBERTa V2 tokenizer
        tokenizer = BertWordPieceTokenizer(lowercase=False)
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
        ])
    else:
        raise ValueError(f"Invalid tokenizer type: {args.tokenizer_type}")
    # Save the tokenizer files
    print("Saving tokenizer..")
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_model(os.path.join(save_dir))

    if args.tokenizer_type == 'debertav2':
        # Load the tokenizer into transformers as a fast tokenizer
        slow_tokenizer = BertWordPieceTokenizer(os.path.join(save_dir, 'vocab.txt'),
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            mask_token="[MASK]")
        slow_tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))

        tokenizer = DebertaV2TokenizerFast.from_pretrained(
            save_dir,
            tokenizer_file=os.path.join(save_dir,"tokenizer.json"),
            vocab_file=os.path.join(save_dir,"vocab.txt"),
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )

        # Now save the tokenizer properly for Hugging Face compatibility
        print("Saving DeBertaV2tokenizer..")
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
