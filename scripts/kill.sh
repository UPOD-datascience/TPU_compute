#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Stopping all running processes..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="sudo pkill -f python3"
