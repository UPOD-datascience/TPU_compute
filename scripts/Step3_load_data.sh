#!/bin/bash
set -o allexport
source ../.env
set +o allexport

echo "Copying models to TPU"
if [[ ${LOCAL_MODEL_DIRECTORY} == gs://* ]]; then
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all --command="mkdir -p ~/models && gsutil cp ${LOCAL_MODEL_DIRECTORY}/*.py ~/models/"
else
  gcloud compute tpus tpu-vm scp \
      --recurse \
      --zone="${ZONE}" \
      --project="${PROJECT_ID}" \
      --worker=all \
      ${LOCAL_MODEL_DIRECTORY}/*.py ${TPU_NAME}:~/models
fi
