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


def clean_text(text):
    '''
    - remove spurious repetitions of characters, punctuation whitespace and linebreaks
    - remove spurious repetitions of words
    '''
    # TODO: Add more/improve cleaning steps as needed
    re_spurious_chars = re.compile(r'(\s)\1{4,}')
    re_spurious_words = re.compile(r'(\b\w+\b)\1{4,}')

    # replace spurious repetitions of characters, punctuation, whitespace and linebreaks with a single instance
    text = re_spurious_chars.sub(r'\1', text)
    text = re_spurious_words.sub(r'\1', text)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='~/tokenizer', required=True)
    parser.add_argument("--vocab_size", type=int, default=50_368)
    parser.add_argument("--min_frequency", type=int, default=100)
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

    # Combine texts from all corpus files
    all_texts = []
    print("Reading files..")
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
    tokenizer.save_model(save_dir)

    if args.tokenizer_type == 'debertav2':
        # Load the tokenizer into transformers as a fast tokenizer
        tokenizer = DebertaV2TokenizerFast.from_pretrained(save_dir)
        print("Saving DeBertaV2tokenizer..")
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_model(save_dir)


if __name__ == "__main__":
    main()
