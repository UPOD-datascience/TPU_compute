#!/bin/bash

# Source environment variables from .env file
set -o allexport
source ~/.env
set +o allexport

# Add before running python script
#export TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false --tf_xla_async_compilation=false"
#export PYTORCH_XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_gpu_autotune_level=0"

while true; do
  echo "NeoBERT training with GCS parquet dataset started at $(date)."
  python3 /home/${USERNAME}/models/train_tpu_neobert_gcs.py \
    --model_dir=${MODEL_BUCKET}/neobert_model \
    --tokenizer_path=${NEOBERT_TOKENIZER_PATH} \
    --config_path=${NEOBERT_CONFIG_PATH} \
    --dataset_path=${DATASET_PATH} \
    --validation_dataset_path=${VALIDATION_DATASET_PATH} \
    --text_column=${TEXT_COLUMN} \
    --batch_size=${BATCH_SIZE} \
    --vocab_size=${VOCAB_SIZE} \
    --max_length=${MAX_SEQ_LEN} \
    --mlm_probability=${MLM_PROB} \
    --pad_to_multiple=8 \
    $([ "${PACK_SEQUENCES}" = "true" ] && echo "--pack_sequences") \
    $([ "${MASK_ALL}" = "true" ] && echo "--mask_all") \
    --learning_rate=${LR} \
    --decay_rate=${WEIGHT_DECAY} \
    --num_epochs=${NUM_EPOCHS} \
    --log_every=${LOGGING_STEPS} \
    --warmup_ratio=${WARMUP_RATIO} \
    --grad_clip=${GRAD_CLIP} \
    --seed=${SEED} \
    $([ "${STREAMING}" = "true" ] && echo "--streaming") \
    --shuffle_buffer_size=${SHUFFLE_BUFFER_SIZE} \
    --steps_per_epoch=${STEPS_PER_EPOCH} \
    --local_dataset_cache=${TMP_DIR}/dataset_cache \
    --validation_cache_path=${TMP_DIR}/validation_cache \
    --validation_steps=${VALIDATION_STEPS} \
    --eval_every_n_epochs=${EVAL_EVERY_N_EPOCHS} \
    $([ -n "${MODEL_CHECKPOINT}" ] && [ "${CONTINUE_FROM_CHECKPOINT}" = "true" ] && echo "--pretrained_model_path=${MODEL_CHECKPOINT}") \
    2>&1 | tee -a ~/logs.txt

  EXIT_CODE=$?
  echo "NeoBERT training exited with code ${EXIT_CODE} at $(date). Restarting in 60 seconds..." | tee -a ~/logs.txt
  sleep 60
done
