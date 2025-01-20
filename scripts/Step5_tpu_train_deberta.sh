#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Setting environment variables..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PJRT_DEVICE=TPU && export XLA_USE_PJRT=1 && export TPU_NAME=${TPU_NAME} && export OMP_NUM_THREADS=1 && export HF_DATASETS_VERBOSITY=debug && export TOKENIZERS_PARALLELISM=false"

echo "Removing dataset cache folders for HuggingFace"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="rm -rf /home/bes3/.cache/huggingface/datasets/json/*"

echo "Stopping all running processes..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="sudo pkill -f python3"

echo "Starting training..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="python3 /home/${USERNAME}/models/train_deberta.py  \
  --dataset_dir=${LOCAL_DATA} \
  --tmp_dir=${TMP_DIR} \
  --output_dir=${MODEL_BUCKET} \
  --tokenizer_name_or_path=/home/${USERNAME}/tokenizer \
  --per_device_train_batch_size=16 \
  --max_seq_length=512 \
  --num_cores=8 \
  --pre_tokenized \
  --learning_rate=0.0001 \
  --weight_decay=0.001 \
  --streaming_data \
  --num_train_epochs=20 2>&1 | tee ~/logs.txt"

# ideally you would launch a shell script on the workers like
# nohup some_script.sh & exit
# so you can disconnect from the ssh session and the script will keep running
# important to have dashboard running to keep track of progress
