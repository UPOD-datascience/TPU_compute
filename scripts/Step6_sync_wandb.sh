#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Syncing wandb..."
nohup gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=0 \
  --command="wandb sync /home/bes3/temp"
