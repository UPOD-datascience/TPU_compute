#!/usr/bin/env python3
"""
Simple TPU device test script to verify XLA tensor operations work correctly.
This script helps debug TPU tensor placement issues.
"""

import torch
import os
import sys

try:
    from torch_xla.runtime import world_size, global_ordinal
    import torch_xla.core.xla_model as xm
    print("XLA import successful")
except ImportError as e:
    print(f"XLA import failed: {e}")
    sys.exit(1)

def test_basic_xla():
    """Test basic XLA operations"""
    print("=== Basic XLA Test ===")
    
    # Get XLA device
    device = xm.xla_device()
    print(f"XLA device: {device}")
    print(f"Global ordinal: {global_ordinal()}")
    print(f"World size: {world_size()}")
    
    # Test tensor creation and movement
    print("\n--- Tensor Creation Test ---")
    
    # Create CPU tensor
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f"CPU tensor: {cpu_tensor}, device: {cpu_tensor.device}")
    
    # Move to XLA device
    xla_tensor = cpu_tensor.to(device)
    print(f"XLA tensor: {xla_tensor}, device: {xla_tensor.device}")
    print(f"Is XLA tensor: {'xla' in str(xla_tensor.device).lower()}")
    
    # Test dtype conversion
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    xla_bfloat16 = float_tensor.to(device, dtype=torch.bfloat16)
    print(f"XLA bfloat16 tensor: {xla_bfloat16}, device: {xla_bfloat16.device}, dtype: {xla_bfloat16.dtype}")
    
    # Test basic operations
    print("\n--- Basic Operations Test ---")
    a = torch.tensor([1.0, 2.0, 3.0]).to(device)
    b = torch.tensor([4.0, 5.0, 6.0]).to(device)
    c = a + b
    print(f"Addition result: {c}, device: {c.device}")
    
    # Force synchronization
    xm.mark_step()
    print("XLA synchronization completed")
    
    return True

def test_embedding_like_operation():
    """Test embedding-like operations that often cause issues"""
    print("\n=== Embedding-like Operation Test ===")
    
    device = xm.xla_device()
    
    # Create embedding-like tensor (vocabulary x embedding_dim)
    vocab_size = 1000
    embed_dim = 128
    embedding_weight = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16).to(device)
    print(f"Embedding weight shape: {embedding_weight.shape}, device: {embedding_weight.device}")
    
    # Create input ids
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    print(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}, dtype: {input_ids.dtype}")
    
    # Test embedding lookup
    try:
        embeddings = torch.embedding(embedding_weight, input_ids)
        print(f"Embedding lookup successful! Output shape: {embeddings.shape}, device: {embeddings.device}")
        xm.mark_step()
        return True
    except Exception as e:
        print(f"Embedding lookup failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing similar to training loop"""
    print("\n=== Batch Processing Test ===")
    
    device = xm.xla_device()
    
    # Simulate a batch like from DataLoader
    batch = {
        'input_ids': torch.randint(0, 1000, (2, 10)),
        'attention_mask': torch.ones(2, 10),
        'labels': torch.randint(0, 1000, (2, 10))
    }
    
    print("Original batch:")
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
    
    # Process batch like in training loop
    processed_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float32:
                processed_batch[k] = v.to(device, dtype=torch.bfloat16)
            else:
                processed_batch[k] = v.to(device)
        else:
            processed_batch[k] = v
    
    print("\nProcessed batch:")
    for k, v in processed_batch.items():
        if isinstance(v, torch.Tensor):
            is_xla = 'xla' in str(v.device).lower()
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, is_xla={is_xla}")
    
    xm.mark_step()
    return True

def test_model_like_operation():
    """Test a simple model-like forward pass"""
    print("\n=== Model-like Operation Test ===")
    
    device = xm.xla_device()
    
    # Create simple linear layer
    input_dim = 128
    output_dim = 64
    linear_weight = torch.randn(output_dim, input_dim, dtype=torch.bfloat16).to(device)
    linear_bias = torch.randn(output_dim, dtype=torch.bfloat16).to(device)
    
    print(f"Linear weight: {linear_weight.shape}, device: {linear_weight.device}")
    print(f"Linear bias: {linear_bias.shape}, device: {linear_bias.device}")
    
    # Create input
    batch_size = 4
    input_tensor = torch.randn(batch_size, input_dim, dtype=torch.bfloat16).to(device)
    print(f"Input tensor: {input_tensor.shape}, device: {input_tensor.device}")
    
    # Forward pass
    try:
        output = torch.nn.functional.linear(input_tensor, linear_weight, linear_bias)
        print(f"Forward pass successful! Output: {output.shape}, device: {output.device}")
        xm.mark_step()
        return True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("TPU Device Test Script")
    print("=" * 50)
    
    # Check environment
    print("Environment variables:")
    print(f"  TPU_NUM_DEVICES: {os.environ.get('TPU_NUM_DEVICES', 'Not set')}")
    print(f"  TPU_CHIPS_PER_HOST_BOUNDS: {os.environ.get('TPU_CHIPS_PER_HOST_BOUNDS', 'Not set')}")
    print(f"  TPU_WORKER_ID: {os.environ.get('TPU_WORKER_ID', 'Not set')}")
    print()
    
    tests = [
        ("Basic XLA", test_basic_xla),
        ("Embedding-like Operation", test_embedding_like_operation),
        ("Batch Processing", test_batch_processing),
        ("Model-like Operation", test_model_like_operation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            success = test_func()
            results[test_name] = success
            print(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"{test_name}: FAILED with exception: {e}")
            results[test_name] = False
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:25}: {status}")
        if not success:
            all_passed = False
    
    print("\nOverall:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())