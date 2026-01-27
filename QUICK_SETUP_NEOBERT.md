# Quick Setup Guide: NeoBERT with GCS Parquet Datasets

## Overview
This guide helps you quickly set up NeoBERT training on TPU using parquet files stored in Google Cloud Storage.

## Prerequisites
1. Google Cloud SDK installed and authenticated
2. Access to TPU resources in your GCP project
3. Parquet dataset files uploaded to GCS
4. NeoBERT tokenizer prepared

## Step 1: Configure Environment
Edit `.neobert.env` with your specific settings:

```bash
# Essential settings to verify/update:
PROJECT_ID=your-project-id
ZONE=us-central2-b
TPU_NAME=tpu_trainer_neobert
DATASET_PATH=gs://your-bucket/path/to/parquet/files
VALIDATION_DATASET_PATH=gs://your-bucket/path/to/validation/parquet  # Optional validation set
NEOBERT_TOKENIZER_PATH=/home/bes3/tokenizer
TEXT_COLUMN=text  # Column name in your parquet files

# Model settings (adjust as needed):
BATCH_SIZE=8
MAX_SEQ_LEN=512
VOCAB_SIZE=100000
MLM_PROB=0.15
LR=0.0003
```

## Step 2: Verify Dataset
Before training, verify your GCS dataset:

```bash
cd scripts
./verify_dataset.sh
```

This will:
- ✅ Check GCS connectivity for training and validation datasets
- ✅ Validate parquet file structure  
- ✅ Confirm text column exists in both datasets
- ✅ Provide optimization recommendations

## Step 3: Start Training
Run the complete training pipeline:

```bash
cd scripts
./Full_Run_train.sh
```

This automatically:
1. Creates data disk
2. Uploads data to disk  
3. Starts TPU VM
4. Installs prerequisites
5. Mounts disk
6. Uploads training scripts
7. Starts NeoBERT training

## Key Files Created/Modified

### Training Scripts
- `models/train_tpu_neobert_gcs.py` - Main training script for GCS parquet
- `scripts/train_wrapper_neobert.sh` - NeoBERT-specific wrapper
- `scripts/verify_dataset.sh` - Dataset verification tool

### Configuration  
- `.neobert.env` - Updated for NeoBERT + GCS parquet
- `scripts/Full_Run_train.sh` - Now uses `.neobert.env`
- `scripts/Step5_tpu_train.sh` - Updated for NeoBERT wrapper

## Monitoring Training

### Check TPU Status
```bash
gcloud compute tpus tpu-vm list --zone=us-central2-b
```

### View Training Logs
```bash
gcloud compute tpus tpu-vm ssh tpu_trainer_neobert \
  --zone=us-central2-b \
  --command="tail -f ~/logs.txt"
```

### Check tmux Session
```bash
gcloud compute tpus tpu-vm ssh tpu_trainer_neobert \
  --zone=us-central2-b \
  --command="tmux attach -t train_session"
```

## Troubleshooting

### Dataset Issues
- **No parquet files found**: Check `DATASET_PATH` and `VALIDATION_DATASET_PATH` in `.neobert.env`
- **Text column missing**: Verify `TEXT_COLUMN` matches your data in both training and validation sets
- **GCS access denied**: Ensure proper authentication with `gcloud auth login`

### Training Issues  
- **Out of memory**: Reduce `BATCH_SIZE` or `MAX_SEQ_LEN`
- **NeoBERT import error**: Install with `pip install neobert`
- **TPU not found**: Check `TPU_NAME` and `ZONE` settings

### Performance Optimization
- **Large dataset (>100 files)**: Set `STREAMING=true`
- **Small dataset (<10 files)**: Set `STREAMING=false`
- **Slow loading**: Increase `SHUFFLE_BUFFER_SIZE`
- **Skip validation**: Remove `VALIDATION_DATASET_PATH` to train without validation
- **Frequent validation**: Set `EVAL_EVERY_N_EPOCHS=1` for validation every epoch

## Important Configuration Notes

### Memory Management
- **Streaming mode** (`STREAMING=true`): Uses constant memory, good for large datasets
- **In-memory mode** (`STREAMING=false`): Faster but loads all data into RAM

### Batch Size Guidelines
- Start with `BATCH_SIZE=8` (per-core)
- Reduce if OOM errors occur
- Total effective batch size = `BATCH_SIZE × 8 × GRAD_ACCUM_STEPS`

### Sequence Length
- NeoBERT default: `MAX_SEQ_LEN=512`
- Longer sequences = more memory usage
- Shorter sequences = faster training

## Model Outputs
Training saves to `MODEL_BUCKET`:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights  
- `tokenizer.json` - Tokenizer files
- Training logs with validation metrics in `~/logs.txt` on TPU VM

## Quick Commands Reference

```bash
# Verify training and validation datasets
./scripts/verify_dataset.sh

# Start full training pipeline with validation 
./scripts/Full_Run_train.sh

# Check training logs (includes validation metrics)
gcloud compute tpus tpu-vm ssh tpu_trainer_neobert --zone=us-central2-b --command="tail -f ~/logs.txt"

# Stop training
gcloud compute tpus tpu-vm ssh tpu_trainer_neobert --zone=us-central2-b --command="tmux kill-session -t train_session"

# Delete TPU (to save costs)
gcloud compute tpus tpu-vm delete tpu_trainer_neobert --zone=us-central2-b
```

Happy training! 🚀