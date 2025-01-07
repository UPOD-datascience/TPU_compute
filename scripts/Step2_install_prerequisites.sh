#!/bin/bash
gcloud components update

# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

echo "Checking if TPU ${TPU_NAME} exists..."

# Temporarily disable `set -e` around grep
set +e
gcloud compute tpus tpu-vm list --zone="${ZONE}" --project="${PROJECT_ID}" | grep -q "${TPU_NAME}"
grep_exit_code=$?
set -e

if [ $grep_exit_code -eq 0 ]; then
    echo "TPU ${TPU_NAME} already exists, skipping creation."
else
    echo "Creating TPU ${TPU_NAME}..."
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
      --zone="${ZONE}" \
      --project="${PROJECT_ID}" \
      --accelerator-type="${ACCELERATOR_TYPE}" \
      --version "${RUNTIME_VERSION}"
fi

gcloud compute tpus tpu-vm list --zone=${ZONE} --project=${PROJECT_ID}

echo "Installing libraries..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME}  \
--project ${PROJECT_ID} \
--zone  ${ZONE} \
--worker=all \
--command="
sudo apt-get update
sudo apt-get install libopenblas-dev -y
python -m pip install numpy
python -m pip install einops==0.8.0
python -m pip install mosaicml[nlp,wandb]==0.22.0
python -m pip install mosaicml-streaming==0.7.6
python -m pip install omegaconf==2.3.0
python -m pip install triton==2.3.0
python -m pip installtyping-extensions
python -m pip install google-cloud
python -m pip install google-cloud-storage
"

echo "Installing more libraries..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
python -m pip install install torch
python -m pip install torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html \
python -m pip install flash-attn \
python -m pip install jupyter transformers datasets[gcs] evaluate accelerate tensorboard scikit-learn  --upgrade"

echo "Cloning XLA..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
 --zone=${ZONE} \
 --project=${PROJECT_ID} \
 --worker=all --command="git clone -b r2.5 https://github.com/pytorch/xla.git"
