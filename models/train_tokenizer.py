# train_tokenizer.py
from tokenizers import (
        ByteLevelBPETokenizer,
        SentencePieceBPETokenizer,
        BertWordPieceTokenizer,
        )
from transformers import (
        DebertaV2TokenizerFast,
        PreTrainedTokenizerFast,
        LlamaTokenizerFast,
        LongformerTokenizerFast,
        RobertaTokenizerFast,
        BigBirdTokenizerFast
)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordPiece

import os
import argparse
import json
import re
import io
from google.cloud import storage
import pandas as pd
import ftfy
import pyarrow.parquet as pq
import sentencepiece as spm
import tempfile
import platform
from typing import List, Iterator

# Windows-specific fixes for SentencePiece
if platform.system() == "Windows":
    # Set environment variables to prevent crashes
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Import with error handling
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

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
    parser.add_argument("--tokenizer_type", type=str, choices=['bbpe', 'sentencebpe', 'debertav2', 'bertwordpiece', 'modernbert', 'llama', 'longformer', 'bigbird'], default='bbpe')
    args = parser.parse_args()

    print("Getting file list..")
    if args.data_dir.startswith("gs://"):
        client = storage.Client.from_service_account_json('../gsa.json')
        bucket_name = args.data_dir.split("/")[2]
        prefix = "/".join(args.data_dir.split("/")[3:])
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        corpus_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(('.jsonl', '.json', '.txt', '.parquet'))]
    else:
        corpus_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(('.jsonl', '.json', '.txt', '.parquet'))]

    print("Reading files..")
    # Combine texts from all corpus files
    if not args.iterative:
        all_texts = []
        for file_path in corpus_files:
            if file_path.endswith('.jsonl') | file_path.endswith('.json'):
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
            if file_path.endswith('.jsonl') | file_path.endswith('.json'):
                text_iterators.append(read_jsonl_iterator(file_path))
            elif file_path.endswith('.txt'):
                text_iterators.append(read_txt_iterator(file_path))
            elif file_path.endswith('.parquet'):
                text_iterators.append(read_parquet_iterator(file_path))

        all_texts_iterator = combine_iterators(text_iterators)
        all_texts = map(clean_text, all_texts_iterator)

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.tokenizer_type in ['bbpe','longformer', 'robert']:
        # Initialize a ByteLevel BPE tokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer._tokenizer.post_processor.verbose = True
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "<s>", "</s>", "<pad>", "<unk>", "<mask>"
        ])
        print("Saving tokenizer..")
        tokenizer.save_model(os.path.join(save_dir))
        if args.tokenizer_type == 'longformer':
            tokenizer_fast = LongformerTokenizerFast(merges_file=os.path.join(save_dir, "merges.txt"), vocab_file=os.path.join(save_dir, "vocab.json"))
        else:
            tokenizer_fast = RobertaTokenizerFast(merges_file=os.path.join(save_dir, "merges.txt"), vocab_file=os.path.join(save_dir, "vocab.json"))
        tokenizer_fast.save_pretrained(os.path.join(save_dir))
    elif args.tokenizer_type == 'sentencebpe':
        # Initialize a SentencePiece BPE tokenizer
        tokenizer = SentencePieceBPETokenizer()
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "<s>", "</s>", "<pad>", "<unk>", "<mask>"
        ])
        print("Saving tokenizer..")
        tokenizer.save_model(os.path.join(save_dir))
    elif args.tokenizer_type in  ['bertwordpiece', 'debertav2']:
        # Initialize a DeBERTa V2 tokenizer
        tokenizer = BertWordPieceTokenizer(lowercase=False)
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
        ])
        # Save the tokenizer files
        print("Saving tokenizer..")
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
    elif args.tokenizer_type=='modernbert':
        # Initialize a DeBERTa V2 tokenizer
        tokenizer = BertWordPieceTokenizer(lowercase=False)
        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(all_texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                "<|padding|>",
                "<|endoftext|>",
                "|||IP_ADDRESS|||",
                "|||EMAIL_ADDRESS|||",
                "|||PHONE_NUMBER|||"
        ])
        tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

        # Wrap in PreTrainedTokenizerFast for ease of use:
        tokenizer_fast = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(save_dir, "tokenizer.json"),
            model_max_length=1024,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            additional_special_tokens=["<|padding|>",
                                    "<|endoftext|>",
                                    "|||IP_ADDRESS|||",
                                    "|||EMAIL_ADDRESS|||",
                                    "|||PHONE_NUMBER|||"]
        )
        print("Saving tokenizer..")
        tokenizer_fast.save_pretrained(os.path.join(save_dir))
    elif args.tokenizer_type == 'llama':
        # Initialize a SentencePiece BPE tokenizer (appropriate for Llama)
        tokenizer = SentencePieceBPETokenizer()

        # Llama special tokens
        special_tokens = [
            "<unk>",  # Unknown token
            "<s>",    # Beginning of sentence
            "</s>",   # End of sentence
            "<pad>"   # Padding token
        ]

        # Train the tokenizer
        print("Training tokenizer..")
        tokenizer.train_from_iterator(
            all_texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens
        )

        # Save the basic tokenizer files
        print("Saving base tokenizer..")
        tokenizer.save_model(os.path.join(save_dir))

        # Save the tokenizer.json file for compatibility
        tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

        # Wrap in LlamaTokenizerFast for HuggingFace compatibility
        print("Creating LlamaTokenizerFast..")
        tokenizer_fast = LlamaTokenizerFast(
            tokenizer_file=os.path.join(save_dir, "tokenizer.json"),
            model_max_length=2048,  # Standard context length for Llama models
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>"
        )
        # Save the Llama tokenizer with all the necessary files
        print("Saving LlamaTokenizerFast..")
        tokenizer_fast.save_pretrained(os.path.join(save_dir))
    elif args.tokenizer_type == 'bigbird':
        # First, write all texts to a temporary file for SentencePiece training
        print("Preparing data for SentencePiece training..")
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            temp_filename = temp_file.name
            if args.iterative:
                for text in all_texts:
                    temp_file.write(text + '\n')
            else:
                for text in all_texts:
                    temp_file.write(text + '\n')

        # Train SentencePiece model directly
        print("Training SentencePiece model..")
        spm_model_prefix = os.path.join(save_dir, "sentencepiece")

        # Configure SentencePiece trainer
        spm.SentencePieceTrainer.train(
            input=temp_filename,
            model_prefix=spm_model_prefix,
            vocab_size=args.vocab_size,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            user_defined_symbols=['<mask>'],
            character_coverage=0.9995,
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
            num_threads=1 #os.cpu_count()
        )

        # Clean up temporary file
        os.unlink(temp_filename)

        # Create BigBird tokenizer using the trained SentencePiece model
        print("Creating BigBirdTokenizerFast..")
        tokenizer_fast = BigBirdTokenizerFast(
            vocab_file=f"{spm_model_prefix}.model",
            model_max_length=4096,  # BigBird's extended context length
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
        )

        # Save the BigBird tokenizer with all necessary files
        print("Saving BigBirdTokenizerFast..")
        tokenizer_fast.save_pretrained(os.path.join(save_dir))

    else:
        raise ValueError(f"Invalid tokenizer type: {args.tokenizer_type}")


if __name__ == "__main__":
    main()
