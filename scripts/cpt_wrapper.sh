#!/bin/bash

# Source environment variables from .env file
set -o allexport
source ~/.env
set +o allexport

huggingface-cli login ${HUGGINGFACE_TOKEN}

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
    --save_epoch_percentage=${SAVE_PERCENTAGE} \
    --logging_steps=${LOGGING_STEPS} \
    --num_warmup_steps=${NUM_WARMUP} \
    --num_cores=4 \
    --max_seq_len=${MAX_SEQ_LEN} \
    --learning_rate=${LR} \
    --lr_schedule=cosine \
    --num_cycles=${NUM_CYCLES} \
    --eta_min=${LR_MIN} \
    --mlm_probability=${MLM_PROB} \
    --streaming_data \
    --training_file=${SHUFFLED_DATASET_EXT} \
    --validation_file=${DATASET_FOLDER}/validation/*.${DATASET_FORMAT} \
    --checkpoint_path=${MODEL_CHECKPOINT} \
    --max_steps_per_epoch=${MAX_STEPS_PER_EPOCH} \
    --weight_decay=${WEIGHT_DECAY} \
    --wandb_key=${WANDB_KEY} \
    --huggingface_token=${HUGGINGFACE_TOKEN} \
    --TPU_NAME=${TPU_NAME} \
    --TPU_DISK=${EXT_DISK_NAME} \
    --num_train_epochs=1 2>&1 | tee -a ~/logs.txt

    # ------------------------------------------------------------------
    # CLEANUP: kill any leftover TPU gRPC servers so the next run can
    # rebind to port 8471 without restarting the VM/TPU
    # ------------------------------------------------------------------
    echo "Cleaning up stale TPU servers…" | tee -a ~/logs.txt
    pids=$(lsof -ti TCP:8471)
    if [ -n "$pids" ]; then
      echo "  killing TPU server PIDs: $pids" | tee -a ~/logs.txt
      kill -9 $pids
    else
      echo "  no TPU servers found." | tee -a ~/logs.txt
    fi

  EXIT_CODE=$?
  echo "Training exited with code ${EXIT_CODE} at $(date). Restarting in 60 seconds..." | tee -a ~/logs.txt
  sleep 60
done
