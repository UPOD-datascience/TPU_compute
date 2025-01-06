#!/bin/bash
gcloud components update

# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

gcloud compute tpus tpu-vm list --zone=${ZONE} --project=${PROJECT_ID} | grep -q ${TPU_NAME}

if [ $? -ne 0 ]; then
  gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version ${RUNTIME_VERSION}
else
  echo "TPU ${TPU_NAME} already exists, skipping creation."
fi


echo "Installing libraries..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME}  \
--project ${PROJECT_ID} \
--zone  ${ZONE} \
--worker=all \
--command="
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip install numpy
pip install typing-extensions
pip install google-cloud
pip install google-cloud-storage
"

echo "Installing more libraries..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
pip install torch torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html \
pip install jupyter transformers datasets[gcs] evaluate accelerate tensorboard scikit-learn  --upgrade"

echo "Cloning XLA..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
 --zone=${ZONE} \
 --project=${PROJECT_ID} \
 --worker=all --command="git clone -b r2.5 https://github.com/pytorch/xla.git"
