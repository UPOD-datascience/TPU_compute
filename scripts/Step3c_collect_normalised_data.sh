#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Download, normalize and re-upload the GCS data..locally!"
python3 normalised_dataset_collector.py \
--data_bucket=${DATA_BUCKET} \
--train_loc=${DATA_BUCKET_TRAIN_NORMALISED} \
--validation_loc=${DATA_BUCKET_VAL_NORMALISED} \
--save_dir_local=${LOCAL_DATA} \
--save_dir_gcs=${DATA_BUCKET} \
--write_mode=parquet \
