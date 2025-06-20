# Exit immediately if a command exits with a non-zero status
# set -e

# # Export variables from .env file
# set -o allexport
# source ../.longformer.env
# set +o allexport

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
sudo apt-get update && sudo apt-get install libffi-dev
pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0 -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html"

# echo "Installing libraries..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME}  \
 --project ${PROJECT_ID} \
 --zone  ${ZONE} \
 --worker=all \
 --command="pip install google-cloud-storage"
# pip install google-cloud
# pip install google-cloud-tpu
# pip install google-cloud-storage

echo "Installing more libraries..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
pip install transformers==4.52.4
pip install tokenizers datasets tqdm wandb safetensors nltk huggingface_hub[cli]
pip install accelerate>=0.26.0
pip install sentencepiece
pip install gcsfs==2024.10.0
pip install fsspec==2024.10.0"

echo "Cloning XLA..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
 --zone=${ZONE} \
 --project=${PROJECT_ID} \
 --worker=all --command="git clone -b r2.5 https://github.com/pytorch/xla.git"


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
