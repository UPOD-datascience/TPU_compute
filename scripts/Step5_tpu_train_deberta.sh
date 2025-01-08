#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Starting test..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="
  sudo pkill -f python
  PJRT_DEVICE=TPU python -m torch_xla.distributed.xla_multiprocessing --nprocs=8 ~/models/test.py"

echo "Starting training..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="
  sudo pkill -f python
  PJRT_DEVICE=TPU python -m torch_xla.distributed.xla_multiprocessing --nprocs=8 ~/models/train_deberta.py  \
  --train_file=gs://${DATA_BUCKET_TRAIN} \
  --validation_file=gs://${DATA_BUCKET_VAL} \
  --output_dir=gs://${MODEL_BUCKET} \
  --tokenizer_name_or_path=~/tokenizer \
  --per_device_train_batch_size=16 \
  --max_seq_length=1024 \
  --learning_rate=0.0001 \
  --num_train_epochs=1 2>&1 | tee ~/logs.txt"
