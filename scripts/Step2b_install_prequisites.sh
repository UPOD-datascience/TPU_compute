# Exit immediately if a command exits with a non-zero status
# set -e

# # Export variables from .env file
# set -o allexport
# source ../.llama.env
# set +o allexport

echo "Installing XLA..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="sudo apt-get update && sudo apt-get install libffi-dev libopenmpi-dev"

echo "Installing all python prerequisites and dependencies..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
pip install torch==2.6.0 torch_xla[tpu]==2.6.0 -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html &&
pip install google-cloud-storage &&
pip install transformers==4.52.4 &&
pip install tokenizers datasets tqdm wandb safetensors nltk huggingface_hub[cli] &&
pip install accelerate==0.26.0 &&
pip install deepspeed==0.17.2 &&
pip install sentencepiece &&
pip install mpi4py &&
pip install gcsfs==2024.10.0 &&
pip install fsspec==2024.10.0"

echo "Installing XLA..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
[ ! -d 'xla' ] && git clone -b r2.5 https://github.com/pytorch/xla.git || echo 'XLA directory already exists, skipping clone'
"

echo "All prerequisites installed successfully!"

# echo "Setting permissions..."
# gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
# --zone=${ZONE} \
# --project=${PROJECT_ID} \
# --worker=all --command="
# sudo groupadd accelerator
# sudo usermod -a -G accelerator ${USER}
# sudo chgrp accelerator /dev/accel*
# sudo chmod 660 /dev/accel*
# "
