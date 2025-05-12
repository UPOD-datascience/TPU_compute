#!/bin/bash
set -e

# Export variables from .env file
# set -o allexport
# source ../.env
# set +o allexport

echo "Mounting data disk to TPU VM..."

# Create mount directory if it doesn't exist
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="sudo mkdir -p /mnt/data"

# Mount the data disk (read-only)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="sudo mount -o ro /dev/sdb /mnt/data && echo 'Disk mounted successfully'"

# Create temp directory
# gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
#     --zone=${ZONE} \
#     --project=${PROJECT_ID} \
#     --worker=all \
#     --command="mkdir -p ${TMP_DIR}"

# Check if disk is mounted properly
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="df -h | grep /mnt/data"

echo "Data disk mounted successfully in read-only mode."
