#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Setting environment variables..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PJRT_DEVICE=TPU && export XLA_USE_PJRT=1 && TPU_NAME=${TPU_NAME}"

echo "Removing dataset cache folders for HuggingFace"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="rm -rf /home/bes3/.cache/huggingface/datasets/json/*"

echo "Starting training..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="
python3 /home/${USERNAME}/models/train_deberta.py  \
  --train_file=${DATA_BUCKET_TRAIN} \
  --validation_file=${DATA_BUCKET_VAL} \
  --output_dir=${MODEL_BUCKET} \
  --tokenizer_name_or_path=/home/${USERNAME}/tokenizer \
  --per_device_train_batch_size=16 \
  --max_seq_length=1024 \
  --num_cores=8 \
  --learning_rate=0.0001 \
  --num_train_epochs=1 2>&1 | tee ~/logs.txt"
