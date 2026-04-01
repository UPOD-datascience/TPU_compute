#!/bin/bash
gcloud components update

# # Exit immediately if a command exits with a non-zero status
# set -e

# # Export variables from .env file
# set -o allexport
# source ../.llama.env
# set +o allexport

echo "Checking if TPU ${TPU_NAME} exists..."

# Temporarily disable `set -e` around grep
set +e
gcloud compute tpus tpu-vm list --zone="${ZONE}" --project="${PROJECT_ID}" | grep -q "${TPU_NAME}"
grep_exit_code=$?
set -e

if [ $grep_exit_code -eq 0 ]; then
    echo "TPU ${TPU_NAME} exists, deleting."
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT_ID}" \
        --quiet
fi

echo "Creating TPU ${TPU_NAME}..."
MAX_RETRIES=500
RETRY_INTERVAL=30

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "Attempt ${attempt}/${MAX_RETRIES}..."
    set +e
    if [ "${PRE_EMPTIBLE}" = true ]; then
        gcloud compute tpus tpu-vm create "${TPU_NAME}" \
          --zone="${ZONE}" \
          --project="${PROJECT_ID}" \
          --accelerator-type="${ACCELERATOR_TYPE}" \
          --version "${RUNTIME_VERSION}" \
          --preemptible \
          --data-disk source=projects/${PROJECT_ID}/zones/${ZONE}/disks/${EXT_DISK_NAME},mode=${EXT_DISK_MODE}
    else
        gcloud compute tpus tpu-vm create "${TPU_NAME}" \
          --zone="${ZONE}" \
          --project="${PROJECT_ID}" \
          --accelerator-type="${ACCELERATOR_TYPE}" \
          --version "${RUNTIME_VERSION}" \
          --data-disk source=projects/${PROJECT_ID}/zones/${ZONE}/disks/${EXT_DISK_NAME},mode=${EXT_DISK_MODE}
    fi
    create_exit_code=$?
    set -e

    if [ $create_exit_code -eq 0 ]; then
        echo "TPU ${TPU_NAME} created successfully on attempt ${attempt}."
        break
    fi

    if [ $attempt -eq $MAX_RETRIES ]; then
        echo "Failed to create TPU ${TPU_NAME} after ${MAX_RETRIES} attempts."
        exit 1
    fi

    echo "Creation failed. Retrying in ${RETRY_INTERVAL} seconds..."
    sleep $RETRY_INTERVAL
done

gcloud compute tpus tpu-vm list --zone=${ZONE} --project=${PROJECT_ID}
