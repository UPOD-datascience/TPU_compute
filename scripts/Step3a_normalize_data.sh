#!/bin/bash
# set -o allexport
# source ../.env
# set +o allexport

echo "Download, normalize and re-upload the GCS data..locally!"
echo "Starting with the training data.."
python3 dataset_normalisation.py --input_dir=${DATA_BUCKET}/training --output_dir=${DATA_BUCKET}/training_normalised

echo "Starting with the validation data.."
python3 dataset_normalisation.py --input_dir=${DATA_BUCKET}/validation --output_dir=${DATA_BUCKET}/validation_normalised
