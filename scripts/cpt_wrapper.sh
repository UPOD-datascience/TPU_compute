#!/bin/bash

# Source environment variables from .env file
set -o allexport
source ~/.cpt.env
set +o allexport

huggingface-cli login ${HF_TOKEN}

while true; do
  echo "Training started at $(date)."
  python3 /home/${USERNAME}/models/cpt_${BASE_MODEL}.py \
    --dataset_dir=${DATASET_FOLDER} \
    --dataset_format=${DATASET_FORMAT} \
    --tmp_dir=${TMP_DIR} \
    --output_dir=${MODEL_BUCKET} \
    --model_name=${MODEL_NAME} \
    --tokenizer_name_or_path=${MODEL_NAME} \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
    --save_epoch_percentage=0.05 \
    --logging_steps=100 \
    --num_warmup_steps=${NUM_WARMUP} \
    --num_cores=16 \
    --max_seq_len=${MAX_SEQ_LEN} \
    --learning_rate=${LR} \
    --streaming_data \
    --shuffle_dataset \
    --shuffle_dataset_path=${SHUFFLED_DATASET_PATH} \
    --shuffle_dataset_gc=${SHUFFLED_DATASET_GC} \
    --checkpoint_path=${MODEL_CHECKPOINT} \
    --max_steps_per_epoch=${NUM_EPOCHS} \
    --weight_decay=${WEIGHT_DECAY} \
    --wandb_key=${WANDB_KEY} \
    --num_train_epochs=5 2>&1 | tee -a ~/logs.txt

  EXIT_CODE=$?
  echo "Training exited with code ${EXIT_CODE} at $(date). Restarting in 60 seconds..." | tee -a ~/logs.txt
  sleep 60
done
