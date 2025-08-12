#!/bin/bash

# Example script for running LoRA fine-tuning on Llama-1B for medical texts
# This script shows the recommended hyperparameters for LoRA training

set -e  # Exit on any error

# Configuration
DATASET_DIR="gs://your-bucket/medical-dataset"
TMP_DIR="/tmp/lora_training"
TOKENIZER_NAME="meta-llama/Llama-3.2-1B"
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="gs://your-bucket/lora-llama-medical"
WANDB_KEY="your_wandb_key_here"
HF_TOKEN="your_huggingface_token_here"
TPU_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-name" -H "Metadata-Flavor: Google")
TPU_DISK="your-tpu-disk-name"

# LoRA Configuration - Recommended settings for medical domain adaptation
LORA_R=16                    # Low rank dimension (8, 16, 32, 64)
LORA_ALPHA=32               # Scaling factor (typically 2x lora_r)
LORA_DROPOUT=0.1            # Dropout for LoRA layers
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"  # All linear layers
LORA_BIAS="none"            # Can be "none", "all", or "lora_only"

# Training Configuration - Adjusted for LoRA
LEARNING_RATE=5e-4          # Higher LR for LoRA (typically 2-10x higher than full fine-tuning)
BATCH_SIZE=8                # Adjust based on TPU memory
GRADIENT_ACCUMULATION=4     # Effective batch size = batch_size * gradient_accumulation * num_cores
NUM_EPOCHS=3                # Fewer epochs needed with LoRA
WARMUP_STEPS=500           # Reduced warmup for LoRA
WEIGHT_DECAY=0.01          # Light weight decay
MAX_SEQ_LENGTH=2048        # Adjust based on your medical texts
MAX_GRAD_NORM=1.0          # Lower gradient clipping for LoRA

# Create temporary directory
mkdir -p $TMP_DIR

echo "Starting LoRA fine-tuning of Llama-1B on medical texts..."
echo "LoRA Configuration:"
echo "  - Rank (r): $LORA_R"
echo "  - Alpha: $LORA_ALPHA"
echo "  - Dropout: $LORA_DROPOUT"
echo "  - Target modules: $LORA_TARGET_MODULES"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

# Run the training
python3 TPU_compute/models/lora_llama.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_format "json" \
    --tmp_dir "$TMP_DIR" \
    --tokenizer_name_or_path "$TOKENIZER_NAME" \
    --model_name "$MODEL_NAME" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --num_warmup_steps $WARMUP_STEPS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --logging_steps 100 \
    --save_epoch_percentage 1.0 \
    --num_cores 8 \
    --streaming_data \
    --wandb_key "$WANDB_KEY" \
    --huggingface_token "$HF_TOKEN" \
    --TPU_NAME "$TPU_NAME" \
    --TPU_DISK "$TPU_DISK" \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --lora_bias "$LORA_BIAS"

echo "LoRA training completed!"
echo "Model adapters saved to: $OUTPUT_DIR"
echo ""
echo "To use the trained LoRA model:"
echo "1. Load the base model: meta-llama/Llama-3.2-1B"
echo "2. Load the LoRA adapters from: $OUTPUT_DIR"
echo "3. Use PEFT library to merge or apply adapters"