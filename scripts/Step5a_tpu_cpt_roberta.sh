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
nohup gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="nohup python3 /home/${USERNAME}/models/cpt_roberta.py  \
  --dataset_dir=${DATASET_FOLDER} \
  --tmp_dir=${TMP_DIR} \
  --output_dir=${MODEL_BUCKET} \
  --tokenizer_name_or_path=/home/${USERNAME}/tokenizer \
  --per_device_train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --save_epoch_percentage=0.5 \
  --logging_steps=5 \
  --num_warmup_steps=2000 \
  --num_cores=8 \
  --pre_tokenized \
  --max_seq_length=${MAX_SEQ_LEN} \
  --learning_rate=0.0001 \
  --keep_in_memory \
  --sharded_data \
  --shuffle_buffer_size=10_000 \
  --weight_decay=0.001 \
  --wandb_key=${WANDB_KEY} \
  --num_train_epochs=1 2>&1 | tee ~/logs.txt &" &
disown

# ideally you would launch a shell script on the workers like
# nohup some_script.sh & exit
# so you can disconnect from the ssh session and the script will keep running
# important to have dashboard running to keep track of progress
