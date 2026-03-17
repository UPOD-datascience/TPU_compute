# Removed Variables from .neobert.env

This document lists all variables that were removed from `.neobert.env` during the cleanup process and explains why they were no longer needed for NeoBERT training.

## Variables Removed

### Legacy Dataset Variables (Not Used with GCS Parquet)
- `DATA_BUCKET=gs://dutch_clinical_corpus` - Replaced by `DATASET_PATH`
- `DATA_BUCKET_TRAIN_NORMALISED=gs://dutch_clinical_corpus/dataset/training` - Not used in NeoBERT script
- `DATA_BUCKET_VAL_NORMALISED=gs://dutch_clinical_corpus/dataset/validation` - Replaced by `VALIDATION_DATASET_PATH`
- `DATASET_FOLDER=gs://dutch_clinical_corpus/dataset` - Not used in NeoBERT script
- `DATASET_FORMAT=parquet` - NeoBERT script automatically detects parquet format
- `SHUFFLED_DATASET_PATH=/home/bes3/temp/shuffled.parquet` - Not used with GCS streaming
- `SHUFFLED_DATASET_EXT=/mnt/data/shuffled_14.parquet` - Not used with GCS streaming
- `SHUFFLED_DATASET_GC=gs://dutch_clinical_corpus/dataset/train/shuffled_14.parquet` - Not used with GCS streaming

### Legacy Training Configuration (Not Used by NeoBERT)
- `GRAD_ACCUM_STEPS=1` - NeoBERT handles batching differently
- `NUM_WARMUP=30000` - Replaced by `WARMUP_RATIO=0.05` (more flexible)
- `MAX_STEPS_PER_EPOCH=100000` - Replaced by `STEPS_PER_EPOCH=1000` (more appropriate)
- `SAVE_PERCENTAGE=2000` - Not used by NeoBERT training script
- `CHECKPOINT_HANDLING=start_with_checkpoint` - Not used by NeoBERT script

### Model Architecture Variables (Defined in NeoBERT Config)
- `HIDDEN_SIZE=1024` - Defined in NeoBERT configuration file
- `HIDDEN_LAYERS=24` - Defined in NeoBERT configuration file  
- `INTERMEDIATE_SIZE=4096` - Defined in NeoBERT configuration file
- `NUM_ATTENTION_HEADS=16` - Defined in NeoBERT configuration file

### External Service Variables (Not Used)
- `WANDB_KEY=29c3dd3150a673a67772f6b5ea35d0e5d835b0fa` - WandB integration not implemented in NeoBERT script
- `MODEL_NAME=chandar-lab/NeoBERT` - Replaced by `NEOBERT_CONFIG_PATH`

### Local Development Variables (Not Used on TPU)
- `LOCAL_DATA=/Users/bes3/DATA/tmp/collection` - Not used in TPU training
- `AUTO_TOKENIZER=True` - NeoBERT uses custom tokenizer loading

### Cache Configuration (Simplified)
- `USE_LOCAL_DATASET_CACHE=true` - Always enabled, no need to configure
- `LOCAL_DATASET_CACHE_PATH=/tmp/dataset_cache` - Automatically derived from `TMP_DIR`
- `VALIDATION_CACHE_PATH=/tmp/validation_cache` - Automatically derived from `TMP_DIR`

## Benefits of Cleanup

### 1. Reduced Complexity
- **Before**: 60+ variables with many duplicates and legacy settings
- **After**: 35 variables, all actively used and well-organized

### 2. Better Organization
Variables are now grouped by purpose:
- TPU and GCP Configuration
- Model Training Parameters
- Dataset Configuration
- NeoBERT Specific Configuration
- Storage and Output Configuration
- Infrastructure Configuration

### 3. Eliminated Confusion
- Removed conflicting or duplicate settings
- Removed BigBird-specific variables that don't apply to NeoBERT
- Removed hardcoded paths that are now dynamically generated

### 4. Improved Maintainability
- Easier to understand what each variable does
- Clearer separation between different configuration areas
- Reduced chance of configuration errors

## Variables That Remain Active

All remaining variables in `.neobert.env` are actively used by:
- TPU creation and management scripts
- NeoBERT training script (`train_tpu_neobert_gcs.py`)
- Training wrapper script (`train_wrapper_neobert.sh`)
- Infrastructure scripts (disk creation, mounting, etc.)

## Migration Notes

If you need to revert any of these changes:
1. The original `.bigbird.env` file contains all the original variables
2. Most removed variables were legacy settings that don't affect NeoBERT training
3. Model architecture variables can be configured in the NeoBERT config file if needed
4. WandB integration can be added back to the training script if required

## Validation

After cleanup, run the verification script to ensure everything works:
```bash
cd scripts
./verify_dataset.sh
```

This confirms that all necessary variables are present and correctly configured.