#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Starting training..."
nohup python3 ../models/cpt_roberta_gpu.py  \
  --dataset_dir=${DATASET_FOLDER} \
  --tmp_dir=${TMP_DIR} \
  --output_dir=${} \
  --model_name=${MODEL_NAME} \
  --tokenizer_name_or_path=${TOKENIZER_PATH}\
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=20 \
  --save_epoch_percentage=0.5 \
  --logging_steps=5 \
  --num_warmup_steps=2000 \
  --num_cores=8 \
  --pre_tokenized \
  --max_seq_length=${MAX_SEQ_LEN} \
  --learning_rate=0.0001 \
  --keep_in_memory \
  --sharded_data \
  --shuffle_buffer_size=10_000 \
  --weight_decay=0.001 \
  --wandb_key=${WANDB_KEY} \
  --num_train_epochs=5 &
disown
