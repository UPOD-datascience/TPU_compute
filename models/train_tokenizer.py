# train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer
import os
import argparse
import json
import re

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

    return text

# Function to read JSONL file and extract text data
def read_jsonl(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data is not None:
                texts.append(data['text'])  # Adjust the key based on your JSON structure
    return texts

# Function to read TXT file and extract text data
def read_txt(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, required=True)
    args = parser.parse_args()

    # Paths to your corpus text files
    corpus_files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)]


    # Combine texts from all corpus files
    all_texts = []
    for file_path in corpus_files:
        if file_path.endswith('.jsonl'):
            all_texts.extend(read_jsonl(file_path))
        elif file_path.endswith('.txt'):
            all_texts.extend(read_txt(file_path))

    # Initialize a ByteLevel BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()

    tokenizer._tokenizer.post_processor.verbose = True
    # Train the tokenizer
    tokenizer.train_from_iterator(all_texts, vocab_size=50_000, min_frequency=100, special_tokens=[
        "<s>", "</s>", "<pad>", "<unk>", "<mask>"
    ])

    # Save the tokenizer files
    save_dir = "./mytokenizer"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_model(save_dir)

if __name__ == "__main__":
    main()
