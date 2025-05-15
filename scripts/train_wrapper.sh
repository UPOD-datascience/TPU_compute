#!/bin/bash

# Source environment variables from .env file
set -o allexport
source ~/.env
set +o allexport

while true; do
  echo "Training started at $(date)."
  python3 /home/${USERNAME}/models/train_${BASE_MODEL}.py \
    --dataset_dir=${DATASET_FOLDER} \
    --dataset_format=${DATASET_FORMAT} \
    --tmp_dir=${TMP_DIR} \
    --output_dir=${MODEL_BUCKET} \
    --model_name=${MODEL_NAME} \
    --tokenizer_name_or_path=/home/${USERNAME}/tokenizer \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
    --save_epoch_percentage=${SAVE_PERCENTAGE} \
    --logging_steps=${LOGGING_STEPS} \
    --num_warmup_steps=${NUM_WARMUP} \
    --num_cores=16 \
    --hidden_size=${HIDDEN_SIZE} \
    --intermediate_size=${INTERMEDIATE_SIZE} \
    --num_hidden_layers=${HIDDEN_LAYERS} \
    --num_attention_heads=${NUM_ATTENTION_HEADS} \
    --max_seq_length=${MAX_SEQ_LEN} \
    --learning_rate=${LR} \
    --streaming_data \
    --shuffle_dataset \
    --shuffle_dataset_path=${SHUFFLED_DATASET_PATH} \
    --shuffle_dataset_ext=${SHUFFLED_DATASET_EXT} \
    --checkpoint_path=${MODEL_CHECKPOINT} \
    --max_steps_per_epoch=${NUM_EPOCHS} \
    --weight_decay=${WEIGHT_DECAY} \
    --wandb_key=${WANDB_KEY} \
    --num_train_epochs=5 2>&1 | tee -a ~/logs.txt

  EXIT_CODE=$?
  echo "Training exited with code ${EXIT_CODE} at $(date). Restarting in 60 seconds..." | tee -a ~/logs.txt
  sleep 60
done
