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

echo "Preloading data into the dataset folder"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="
  python3 /home/${USERNAME}/models/preloading_dataset.py  \
  --train_loc=${DATA_BUCKET_TRAIN} \
  --validation_loc=${DATA_BUCKET_VAL} \
  --save_dir=${LOCAL_DATA} \
  --tokenizer_name_or_path=/home/${USERNAME}/tokenizer \
  --max_seq_length=512"
