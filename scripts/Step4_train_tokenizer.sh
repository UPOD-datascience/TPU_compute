#!/bin/bash
set -o allexport
source ../.env
set +o allexport

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="python3 /home/${USERNAME}/models/train_tokenizer.py  --data_dir=${DATA_BUCKET} --output_dir=/home/${USERNAME}/tokenizer
  "
