#!/bin/bash
gcloud components update

# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

gcloud compute tpus tpu-vm create ${TPU_NAME} \
--zone=${ZONE} \
--project=${PROJECT_ID} \
--accelerator-type=${ACCELERATOR_TYPE} \
--version ${RUNTIME_VERSION}

gcloud compute tpus tpu-vm list --zone=${ZONE} --project=${PROJECT_ID}

gcloud compute tpus tpu-vm ssh ${TPU_NAME}  \
--project ${PROJECT_ID} \
--zone  ${ZONE} \
--worker=all \
--command="
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip3 install numpy
pip3 install typing-extensions
pip install torch torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
pip3 install jupyter transformers datasets[gcs] evaluate accelerate tensorboard scikit-learn  --upgrade"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
 --zone=${ZONE} \
 --project=${PROJECT_ID} \
 --worker=all --command="git clone -b r2.5 https://github.com/pytorch/xla.git"
