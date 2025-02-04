#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Preloading data into the dataset folder"
python3 preloading_dataset.py  \
  --data_bucket=${DATA_BUCKET} \
  --train_loc=${DATA_BUCKET_TRAIN_NORMALISED} \
  --validation_loc=${DATA_BUCKET_VAL_NORMALISED} \
  --save_dir_local=${LOCAL_DATA} \
  --save_dir_gcs=${DATA_BUCKET} \
  --autotokenizer=${AUTO_TOKENIZER} \
  --tokenizer_name_or_path=${TOKENIZER_PATH} \
  --max_seq_length=${MAX_SEQ_LEN} \
  --write_mode=jsonl \
