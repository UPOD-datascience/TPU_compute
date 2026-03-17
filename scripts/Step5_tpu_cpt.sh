#!/bin/bash
set -o allexport
source ../.llama.env
set +o allexport


ENV=".llama.env"

echo "Setting environment variables..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PJRT_DEVICE=TPU && export XLA_USE_PJRT=1 && export TPU_NAME=${TPU_NAME} && export OMP_NUM_THREADS=1 && export HF_DATASETS_VERBOSITY=debug && export TOKENIZERS_PARALLELISM=false;rm -rf /home/${USERNAME}/.cache/*; rm ~/logs.txt;rm -rf /home/${USERNAME}/.cache/*; rm ~/logs.txt"

echo "Uploading ${ENV} as .env and training wrapper script... (TPU_NAME=${TPU_NAME}, ZONE=${ZONE}, PROJECT_ID=${PROJECT_ID}, USERNAME=${USERNAME})"
# Copy .longformer.env to .env on the remote
gcloud compute tpus tpu-vm scp ../${ENV} ${TPU_NAME}:/home/${USERNAME}/.env \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all

echo "Uploading .env and training wrapper script..."
gcloud compute tpus tpu-vm scp cpt_wrapper.sh ${TPU_NAME}:/home/${USERNAME}/ \
    --zone=${ZONE} --project=${PROJECT_ID} --worker=all

echo "Uploading model training script..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="mkdir -p /home/${USERNAME}/models"
gcloud compute tpus tpu-vm scp ../models/cpt_${BASE_MODEL}.py ${TPU_NAME}:/home/${USERNAME}/models/ \
    --zone=${ZONE} --project=${PROJECT_ID} --worker=all

# echo "Killing previous persistent training session..."
# gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
#     --zone=${ZONE} \
#     --project=${PROJECT_ID} \
#     --worker=all \
#     --command="tmux kill-session -t train_session 2>/dev/null; echo 'tmux session killed (or not found)';pkill -f python3; echo 'Python processes killed (or none found)'"

# echo "Stopping all running Python processes..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pkill -f python3; echo 'Python processes killed (or none found)'"

echo "Waiting for processes to fully terminate..."
sleep 10


echo "Cleaning up stale TPU gRPC servers on port 8471...\
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \\    --zone=${ZONE} \\    --project=${PROJECT_ID} \
    --worker=all \
    --command='pids=$(lsof -ti TCP:8471 2>/dev/null); if [ -n "$pids" ]; then echo "Killing TPU server PIDs: $pids\; sudo kill -9 $pids || echo "Failed to kill some processes, continuing...\; else echo "No stale TPU servers found."; fi'
echo "Waiting for TPU devices to be released..."
sleep 10

echo "Verifying no stale processes remain...\
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command='procs=$(pgrep -f "python3|libtpu" 2>/dev/null); if [ -n "$procs" ]; then echo "WARNING: stale processes still running: $procs"; sudo kill -9 $procs; sleep 3; else echo "All clear."; fi' || true

echo "Launching persistent training session..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="chmod +x ~/cpt_wrapper.sh && tmux new-session -d -s train_session '~/cpt_wrapper.sh'"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="tmux ls"

echo "Training session launched successfully!"
