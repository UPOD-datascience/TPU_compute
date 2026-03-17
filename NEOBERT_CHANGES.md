# NeoBERT Configuration Changes Summary

## Overview
This document summarizes the changes made to configure the TPU training system for NeoBERT instead of BigBird.

## Files Modified

### 1. `.neobert.env` - Main Configuration File
**Key Changes:**
- `TPU_NAME=tpu_trainer_neobert` (was: `tpu_trainer_bigbird`)
- `BASE_MODEL=tpu_neobert` (was: `bigbird`) - Critical for script selection
- `MLM_PROB=0.15` (was: `0.2`) - NeoBERT default masking probability
- `BATCH_SIZE=8` (was: `1`) - Per-core batch size for NeoBERT
- `LR=0.0003` (was: `0.0001`) - NeoBERT default learning rate (3e-4)
- `WEIGHT_DECAY=0.01` (was: `0.0001`) - NeoBERT default weight decay
- `MAX_SEQ_LEN=512` (was: `4096`) - NeoBERT typical sequence length
- `MODEL_NAME=chandar-lab/NeoBERT` (was: `google/bigbird-roberta-base`)
- `EXT_DISK_NAME=data-disk-train-neobert` (was: `data-disk-train-bigbird`)
- `LOGGING_STEPS=50` (was: `100`) - NeoBERT script default

**Added NeoBERT-specific parameters:**
- `VOCAB_SIZE=100000` - NeoBERT vocabulary size
- `NEOBERT_CONFIG_PATH=chandar-lab/NeoBERT` - Config path for NeoBERT
- `NEOBERT_TOKENIZER_PATH=/home/bes3/tokenizer` - Path to NeoBERT tokenizer
- `DATASET_NAMES=bookcorpus,wikipedia` - HuggingFace dataset names
- `TEXT_COLUMN=text` - Column name for text data
- `NUM_EPOCHS=3` - Number of training epochs
- `WARMUP_RATIO=0.05` - Warmup ratio
- `GRAD_CLIP=1.0` - Gradient clipping
- `SEED=42` - Random seed
- `PACK_SEQUENCES=true` - Enable sequence packing
- `MASK_ALL=false` - Use standard 80/10/10 masking
- `STREAMING=true` - Enable streaming datasets
- `SHUFFLE_BUFFER_SIZE=10000` - Shuffle buffer size
- `STEPS_PER_EPOCH=1000` - Steps per epoch for streaming

### 2. `scripts/Full_Run_train.sh`
**Changes:**
- `source ../.neobert.env` (was: `source ../.bigbird.env`)

### 3. `scripts/Step5_tpu_train.sh`
**Changes:**
- `source ../.neobert.env` (was: `source ../.bigbird.env`)
- `ENV=".neobert.env"` (was: `ENV=".bigbird.env"`)
- Uses `train_wrapper_neobert.sh` instead of `train_wrapper.sh`

### 4. `models/train_tpu_neobert_gcs.py` - NEW FILE  
**Purpose:**
- Modified NeoBERT training script that supports loading parquet files from GCS
- Downloads parquet files locally and creates streaming or in-memory datasets
- Uses gsutil for GCS file operations

### 5. `scripts/train_wrapper_neobert.sh` - NEW FILE
**Purpose:** 
- NeoBERT-specific training wrapper that calls `train_tpu_neobert_gcs.py` with correct arguments
- Replaces the generic `train_wrapper.sh` which was designed for other models

**Key Features:**
- Maps environment variables to NeoBERT script parameters
- Handles boolean flags properly (pack_sequences, mask_all, streaming)
- Supports conditional checkpoint loading
- Maintains restart loop for fault tolerance

### 6. `scripts/verify_dataset.sh` - NEW FILE
**Purpose:**
- Verification script to check GCS parquet dataset accessibility and structure
- Validates that parquet files exist and contain the expected text column
- Provides recommendations for streaming vs in-memory loading

## Critical Notes

### 1. Training Script Compatibility
The existing `train_wrapper.sh` is **incompatible** with `train_tpu_neobert.py` because:
- Different command-line argument structure
- Different parameter names and types
- NeoBERT script uses PyTorch XLA directly, not the Transformers Trainer

### 2. GCS Dataset Support
The training now supports parquet files stored in Google Cloud Storage:
- Parquet files are automatically downloaded and cached locally
- Supports both streaming and in-memory loading modes
- Uses gsutil for GCS operations (must be installed and authenticated)
- Supports separate validation dataset from single or multiple parquet files
- Validation runs periodically during training for monitoring progress

### 3. Required Dependencies
Ensure the following packages are installed on the TPU VM:
```bash
pip install neobert pandas gsutil
```

Or if using a custom NeoBERT source, set `--neobert_src` parameter.

### 4. Tokenizer Requirements
- The tokenizer path should point to a NeoBERT-compatible tokenizer
- Default vocab size is 100,000 (configurable via `VOCAB_SIZE`)
- Ensure the tokenizer directory exists at `NEOBERT_TOKENIZER_PATH`

### 5. Dataset Configuration  
- `DATASET_PATH` should point to GCS directory containing training parquet files
- `VALIDATION_DATASET_PATH` should point to GCS directory/file containing validation parquet files (optional)
- Format: `gs://bucket-name/path/to/parquet/files`
- Parquet files must contain a text column (configurable via `TEXT_COLUMN`)
- Supports both streaming and in-memory loading modes
- Local caching improves performance on subsequent runs
- Validation dataset is typically loaded in-memory for faster evaluation

### 6. Model Checkpoints
- Model checkpoints are saved as `pytorch_model.bin` (PyTorch format)
- Different from BigBird which might use `model.safetensors`
- Update `MODEL_CHECKPOINT` path accordingly for resuming training

## Validation Checklist

Before running training, verify:
- [ ] NeoBERT package is installed on TPU VM
- [ ] gsutil is installed and authenticated for GCS access
- [ ] Tokenizer exists at specified path
- [ ] GCS parquet files are accessible at `DATASET_PATH`
- [ ] GCS validation parquet files are accessible at `VALIDATION_DATASET_PATH` (if using validation)
- [ ] Parquet files contain the specified `TEXT_COLUMN`
- [ ] Model output directory has write permissions
- [ ] TPU VM has sufficient disk space for model checkpoints and dataset cache
- [ ] Environment variables are properly exported

Run the dataset verification script first:
```bash
cd scripts
chmod +x verify_dataset.sh
./verify_dataset.sh
```

## Usage

Run the complete training pipeline with:
```bash
cd scripts
./Full_Run_train.sh
```

This will:
1. Create the data disk
2. Upload data to disk
3. Start TPU VM
4. Install prerequisites
5. Mount disk
6. Upload training scripts
7. Start NeoBERT training with GCS parquet dataset support

## Dataset Verification

Before training, verify your GCS dataset:
```bash
cd scripts
./verify_dataset.sh
```

This will:
- Check GCS connectivity and file accessibility for training and validation datasets
- Validate parquet file structure for both datasets
- Confirm text column exists in training and validation files
- Provide optimization recommendations
- Estimate appropriate STEPS_PER_EPOCH value
- Verify validation dataset configuration if provided