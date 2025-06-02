#!/usr/bin/env python3
"""
Example script for training a BigBirdTokenizerFast with SentencePiece encoder.
This demonstrates how to use the train_bigbird_tokenizer function.
"""

import os
from train_tokenizer import train_bigbird_tokenizer, read_txt, read_jsonl, clean_text

def main():
    # Example 1: Train from a list of sample texts
    print("Example 1: Training from sample texts")
    sample_texts = [
        "This is a sample text for training the BigBird tokenizer.",
        "BigBird uses SentencePiece as its underlying tokenization algorithm.",
        "The tokenizer will learn subword units from the training corpus.",
        "Make sure to provide enough diverse text data for good tokenization.",
        "The BigBird model supports long sequences up to 4096 tokens."
    ] * 1000  # Repeat to have more training data
    
    # Train the tokenizer
    tokenizer = train_bigbird_tokenizer(
        texts=sample_texts,
        vocab_size=1000,  # Small vocab for demo
        save_dir="./bigbird_tokenizer_sample",
        model_max_length=512
    )
    
    # Test the tokenizer
    test_text = "This is a test sentence for the BigBird tokenizer."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()
    
    # Example 2: Train from text files
    print("Example 2: Training from text files")
    
    # Create sample data directory and files for demonstration
    data_dir = "./sample_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample text files
    with open(os.path.join(data_dir, "sample1.txt"), "w") as f:
        f.write("Natural language processing is a subfield of artificial intelligence.\n")
        f.write("It focuses on the interaction between computers and human language.\n")
        f.write("BigBird is a transformer model that can handle long sequences efficiently.\n")
    
    with open(os.path.join(data_dir, "sample2.txt"), "w") as f:
        f.write("Machine learning models require large amounts of training data.\n")
        f.write("Tokenization is the process of converting text into tokens.\n")
        f.write("SentencePiece is a popular tokenization library for neural networks.\n")
    
    # Read texts from files
    all_texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            texts = read_txt(file_path)
            all_texts.extend(texts)
    
    # Train tokenizer from file data
    tokenizer_from_files = train_bigbird_tokenizer(
        texts=all_texts,
        vocab_size=2000,
        save_dir="./bigbird_tokenizer_files",
        model_max_length=1024
    )
    
    print("Tokenizer trained from files successfully!")
    
    # Example 3: Load a saved tokenizer
    print("Example 3: Loading saved tokenizer")
    from transformers import BigBirdTokenizerFast
    
    loaded_tokenizer = BigBirdTokenizerFast.from_pretrained("./bigbird_tokenizer_sample")
    
    # Test the loaded tokenizer
    test_text = "Loading and testing the saved BigBird tokenizer."
    tokens = loaded_tokenizer.encode(test_text)
    decoded = loaded_tokenizer.decode(tokens)
    print(f"Loaded tokenizer test:")
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()