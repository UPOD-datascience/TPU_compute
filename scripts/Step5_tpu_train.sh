#!/bin/bash
# set -o allexport
# source ../.env
# set +o allexport

ENV=".env"

echo "Setting environment variables... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID})"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PJRT_DEVICE=TPU && export XLA_USE_PJRT=1 && export TPU_NAME=${TPU_NAME} && export OMP_NUM_THREADS=1 && export HF_DATASETS_VERBOSITY=debug && export TOKENIZERS_PARALLELISM=false"

echo "Removing cache folders and logs... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID}, USERNAME=${USERNAME})"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="rm -rf /home/${USERNAME}/.cache/*; rm ~/logs.txt"

    echo "Uploading ${ENV} as .env and training wrapper script... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID}, USERNAME=${USERNAME})"
    # Copy .longformer.env to .env on the remote
    gcloud compute tpus tpu-vm scp ../${ENV} ${TPU_NAME}:/home/${USERNAME}/.env \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all

    # Copy the training wrapper script
    gcloud compute tpus tpu-vm scp train_wrapper.sh ${TPU_NAME}:/home/${USERNAME}/train_wrapper.sh \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all

echo "Stopping all running processes... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID})"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="sudo pkill -f python3"

echo "Killing previous persistent training session... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID})"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="tmux kill-session -t train_session"

echo "Launching persistent training session... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID})"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="chmod +x ~/train_wrapper.sh && tmux new-session -d -s train_session '~/train_wrapper.sh'"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="tmux ls"


echo "Training session launched successfully!"
