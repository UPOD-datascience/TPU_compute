# LoRA Fine-tuning Guide for Llama-1B on Medical Texts

This guide explains how to use Low-Rank Adaptation (LoRA) for fine-tuning Llama-1B models on medical texts to prevent model collapse while maintaining training efficiency.

## Table of Contents

1. [What is LoRA?](#what-is-lora)
2. [Why Use LoRA for Medical Domain Adaptation?](#why-use-lora-for-medical-domain-adaptation)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Configuration](#advanced-configuration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

## What is LoRA?

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that:

- **Reduces trainable parameters**: Instead of updating all model parameters, LoRA only trains small adapter matrices
- **Prevents catastrophic forgetting**: The base model weights remain frozen, preserving general knowledge
- **Enables efficient storage**: Only the small adapter weights need to be saved (~1-10MB vs GBs for full models)
- **Allows multiple adapters**: You can train different LoRA adapters for different tasks/domains

### How LoRA Works

LoRA decomposes weight updates into two low-rank matrices:
- **Original weight**: `W₀` (frozen)
- **LoRA adaptation**: `W₀ + B×A` where `B` and `A` are trainable low-rank matrices
- **Rank**: Controls the capacity of the adaptation (typically 8, 16, 32, or 64)

## Why Use LoRA for Medical Domain Adaptation?

### Prevents Model Collapse
- **Catastrophic forgetting**: Full fine-tuning can cause the model to "forget" general language understanding
- **Domain overfitting**: LoRA maintains the base model's capabilities while adapting to medical terminology
- **Stable training**: Lower risk of training instabilities

### Efficiency Benefits
- **Memory efficient**: ~90% reduction in GPU memory usage
- **Storage efficient**: Adapter weights are typically <50MB vs 2-5GB for full models
- **Training speed**: Faster convergence due to fewer parameters
- **Multiple domains**: Train separate adapters for different medical specialties

### Medical Domain Advantages
- **Preserve general knowledge**: Keep understanding of general language while learning medical terms
- **Ethical safety**: Reduced risk of generating harmful medical advice due to preserved base model behavior
- **Incremental learning**: Easy to add new medical knowledge without retraining everything

## Installation

### 1. Install Dependencies

```bash
pip install -r TPU_compute/requirements_lora.txt
```

### 2. Verify Installation

```python
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM
print("LoRA setup successful!")
```

## Basic Usage

### 1. Prepare Your Training Command

```bash
#!/bin/bash
python3 TPU_compute/models/lora_llama.py \
    --dataset_dir "gs://your-bucket/medical-dataset" \
    --tokenizer_name_or_path "meta-llama/Llama-3.2-1B" \
    --model_name "meta-llama/Llama-3.2-1B" \
    --output_dir "gs://your-bucket/lora-medical-adapter" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --max_seq_length 2048 \
    --wandb_key "your_wandb_key" \
    --huggingface_token "your_hf_token" \
    --streaming_data
```

### 2. Load and Use the Trained Model

```python
from TPU_compute.utils.load_lora_model import load_lora_model, generate_text

# Load model with LoRA adapters
model, tokenizer = load_lora_model(
    base_model_name="meta-llama/Llama-3.2-1B",
    lora_adapter_path="gs://your-bucket/lora-medical-adapter"
)

# Generate medical text
prompt = "The patient presented with chest pain and shortness of breath. The differential diagnosis includes:"
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=200)
print(generated_text)
```

## Advanced Configuration

### LoRA Hyperparameters

| Parameter | Description | Recommended Values | Impact |
|-----------|-------------|-------------------|---------|
| `lora_r` | Rank (adaptation capacity) | 8, 16, 32 | Higher = more capacity, more parameters |
| `lora_alpha` | Scaling factor | 2×lora_r | Controls adaptation strength |
| `lora_dropout` | Dropout rate | 0.05-0.1 | Regularization |
| `lora_target_modules` | Which layers to adapt | All linear layers | More modules = more adaptation |

### Target Modules for Llama

```python
# All attention and MLP layers (recommended)
lora_target_modules = [
    "q_proj",      # Query projection
    "k_proj",      # Key projection  
    "v_proj",      # Value projection
    "o_proj",      # Output projection
    "gate_proj",   # Gate projection (MLP)
    "up_proj",     # Up projection (MLP)
    "down_proj"    # Down projection (MLP)
]

# Attention only (lighter adaptation)
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Custom for medical domain
lora_target_modules = ["q_proj", "v_proj", "gate_proj", "down_proj"]
```

### Learning Rate Guidelines

| Training Type | Learning Rate | Reasoning |
|---------------|---------------|-----------|
| Full Fine-tuning | 1e-5 to 5e-5 | Lower to prevent catastrophic forgetting |
| LoRA | 1e-4 to 1e-3 | Higher because fewer parameters to update |
| Medical LoRA | 3e-4 to 5e-4 | Balanced for domain adaptation |

## Best Practices

### 1. Data Preparation

```python
# Ensure consistent medical text formatting
medical_prompt_template = """### Medical Context:
{context}

### Question:
{question}

### Response:
{response}"""

# Use domain-specific tokenization if needed
# Medical abbreviations, dosages, measurements should be properly tokenized
```

### 2. Training Configuration

```bash
# Recommended settings for medical domain
--use_lora \
--lora_r 16 \                    # Good balance of capacity and efficiency
--lora_alpha 32 \                # 2x the rank
--lora_dropout 0.1 \             # Light regularization
--learning_rate 5e-4 \           # Higher than full fine-tuning
--num_train_epochs 3 \           # Fewer epochs needed
--warmup_steps 500 \             # Reduced warmup
--weight_decay 0.01 \            # Light weight decay
--max_grad_norm 1.0 \            # Lower gradient clipping
--gradient_accumulation_steps 4   # Maintain effective batch size
```

### 3. Monitoring Training

Key metrics to watch:
- **Training loss**: Should decrease steadily
- **Validation perplexity**: Should improve without overfitting
- **Medical accuracy**: Use domain-specific evaluation metrics
- **General capability retention**: Test on general language tasks

### 4. Multiple Adapters Strategy

```python
# Train separate adapters for different medical specialties
specialties = [
    "cardiology",
    "oncology", 
    "neurology",
    "emergency_medicine"
]

# Each adapter can be 16MB vs 2GB for full models
# Load different adapters based on the medical context
```

## Troubleshooting

### Common Issues

#### 1. Training Instability
**Symptoms**: Loss spikes, NaN values
**Solutions**:
- Lower learning rate (try 1e-4 instead of 5e-4)
- Reduce `lora_alpha` 
- Add gradient clipping (`--max_grad_norm 0.5`)
- Increase warmup steps

#### 2. Poor Medical Adaptation
**Symptoms**: Model doesn't learn medical terminology
**Solutions**:
- Increase `lora_r` (try 32 or 64)
- Add more target modules
- Increase training data
- Check data quality and formatting

#### 3. Catastrophic Forgetting
**Symptoms**: Loss of general language capabilities
**Solutions**:
- This shouldn't happen with LoRA, but if it does:
- Lower learning rate
- Reduce number of epochs
- Add general language data to training mix

#### 4. Memory Issues
**Symptoms**: OOM errors
**Solutions**:
- Reduce batch size
- Use gradient checkpointing
- Reduce sequence length
- Use smaller `lora_r`

### TPU-Specific Issues

#### XLA Compilation Errors
```bash
# Set environment variables
export XLA_USE_BF16=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

# Use smaller batch sizes initially
--per_device_train_batch_size 4
```

#### Synchronization Issues
```python
# Ensure all processes wait for model loading
xm.rendezvous("lora_model_loaded")
```

## Performance Optimization

### Memory Optimization

```python
# Enable gradient checkpointing for larger models
model.gradient_checkpointing_enable()

# Use bfloat16 for TPU efficiency
model = model.to(dtype=torch.bfloat16)

# Clear cache regularly
import gc
gc.collect()
```

### Speed Optimization

```bash
# Use streaming datasets for large medical corpora
--streaming_data \
--shuffle_buffer_size 10000

# Optimize batch size for TPU
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4  # 32 effective batch size per TPU

# Use efficient data loading
--num_workers 4
```

### Storage Optimization

```python
# Save only LoRA weights (much smaller)
# Base model: 2.5GB
# LoRA adapter: ~16MB (r=16) to ~64MB (r=64)

# Use safetensors format
model.save_pretrained("./medical_lora", safe_serialization=True)
```

## Evaluation and Validation

### Medical-Specific Evaluation

```python
def evaluate_medical_capabilities(model, tokenizer, test_cases):
    """Evaluate model on medical tasks"""
    results = {
        'medical_terminology': [],
        'diagnostic_reasoning': [],
        'treatment_recommendations': [],
        'safety_responses': []
    }
    
    for case in test_cases:
        response = generate_text(model, tokenizer, case['prompt'])
        # Evaluate response quality
        score = evaluate_medical_response(response, case['expected'])
        results[case['category']].append(score)
    
    return results
```

### General Capability Retention

```python
def test_general_capabilities(model, tokenizer):
    """Ensure model retains general language understanding"""
    general_tasks = [
        "Explain the concept of photosynthesis",
        "Write a short story about friendship", 
        "Solve this math problem: 2x + 5 = 13",
        "Translate 'Hello world' to Spanish"
    ]
    
    for task in general_tasks:
        response = generate_text(model, tokenizer, task)
        print(f"Task: {task}")
        print(f"Response: {response}\n")
```

## Deployment

### Single Adapter Deployment

```python
from peft import PeftModel

# Load for inference
base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base_model, "./medical_lora_adapter")
```

### Multiple Adapter Management

```python
class MedicalLoRAManager:
    def __init__(self, base_model_name):
        self.base_model = LlamaForCausalLM.from_pretrained(base_model_name)
        self.adapters = {}
    
    def load_adapter(self, specialty, adapter_path):
        """Load adapter for specific medical specialty"""
        self.adapters[specialty] = adapter_path
    
    def switch_adapter(self, specialty):
        """Switch to different medical specialty"""
        if specialty in self.adapters:
            return PeftModel.from_pretrained(
                self.base_model, 
                self.adapters[specialty]
            )
        else:
            raise ValueError(f"Adapter for {specialty} not loaded")

# Usage
manager = MedicalLoRAManager("meta-llama/Llama-3.2-1B")
manager.load_adapter("cardiology", "./cardiology_lora")
manager.load_adapter("oncology", "./oncology_lora")

# Switch contexts
cardiology_model = manager.switch_adapter("cardiology")
```

## Conclusion

LoRA provides an excellent solution for adapting Llama models to medical domains while avoiding model collapse. Key benefits:

- **90% fewer trainable parameters**
- **Prevents catastrophic forgetting** 
- **Fast training and inference**
- **Easy to manage multiple specialties**
- **Cost-effective storage and deployment**

Start with the recommended settings in this guide, then fine-tune based on your specific medical dataset and requirements.

For questions or issues, refer to the troubleshooting section or check the PEFT documentation at https://huggingface.co/docs/peft/