"""
Helper utility for loading LoRA-adapted Llama models for inference.
This script demonstrates how to load and use models trained with LoRA adapters.
"""

import torch
from transformers import (
    LlamaTokenizerFast,
    LlamaForCausalLM,
    LlamaConfig
)
from peft import PeftModel, PeftConfig
import argparse
import os
from typing import Optional


def load_lora_model(
    base_model_name: str,
    lora_adapter_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    """
    Load a LoRA-adapted model for inference.
    
    Args:
        base_model_name: Name or path of the base model (e.g., "meta-llama/Llama-3.2-1B")
        lora_adapter_path: Path to the LoRA adapter weights
        device: Device to load the model on ("auto", "cuda", "cpu")
        torch_dtype: Data type for the model weights
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else "auto",
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from: {lora_adapter_path}")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1
):
    """
    Generate text using the LoRA-adapted model.
    
    Args:
        model: The loaded LoRA model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        repetition_penalty: Penalty for repetition
    
    Returns:
        str: Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return only the newly generated part
    return generated_text[len(prompt):]


def merge_and_save_model(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16
):
    """
    Merge LoRA adapters with base model and save the merged model.
    This creates a standalone model that doesn't require PEFT.
    
    Args:
        base_model_name: Name or path of the base model
        lora_adapter_path: Path to the LoRA adapter weights
        output_path: Path to save the merged model
        torch_dtype: Data type for the model weights
    """
    print("Loading model and adapters for merging...")
    
    # Load model and adapters
    model, tokenizer = load_lora_model(
        base_model_name, 
        lora_adapter_path, 
        device="cpu",  # Load on CPU for merging
        torch_dtype=torch_dtype
    )
    
    print("Merging LoRA adapters with base model...")
    
    # Merge adapters into base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # Save merged model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merged model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Load and test LoRA-adapted Llama model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Base model name or path")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA adapter weights")
    parser.add_argument("--prompt", type=str, 
                        default="The patient presented with symptoms of",
                        help="Test prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--merge_and_save", type=str, default=None,
                        help="Path to save merged model (optional)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, tokenizer = load_lora_model(
            args.base_model,
            args.lora_path,
            device=args.device
        )
        
        # Test generation
        print(f"\nTesting generation with prompt: '{args.prompt}'")
        print("-" * 50)
        
        generated = generate_text(
            model, 
            tokenizer, 
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"Generated text:\n{generated}")
        print("-" * 50)
        
        # Merge and save if requested
        if args.merge_and_save:
            merge_and_save_model(
                args.base_model,
                args.lora_path,
                args.merge_and_save
            )
            
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()