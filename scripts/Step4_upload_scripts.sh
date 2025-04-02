#!/bin/bash
# set -o allexport
# source ../.env
# set +o allexport

echo "Copying models to TPU"
if [[ ${LOCAL_MODEL_DIRECTORY} == gs://* ]]; then
  echo "Copying models from GCS"
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all --command="mkdir -p /home/${USERNAME}/models && gsutil cp ${LOCAL_MODEL_DIRECTORY}/*.py /home/${USERNAME}/models/"
else
  echo "Copying models from local"
  gcloud compute tpus tpu-vm scp \
      --recurse \
      --zone="${ZONE}" \
      --project="${PROJECT_ID}" \
      --worker=all \
      ${LOCAL_MODEL_DIRECTORY}/*.py ${TPU_NAME}:/home/${USERNAME}/models
fi

echo "Copying tokenizer to TPU"
if [[ ${LOCAL_TOKENIZER_DIRECTORY} == gs://* ]]; then
  echo "Copying tokenizer from GCS"
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all --command="mkdir -p /home/${USERNAME}/tokenizer && gsutil cp ${LOCAL_TOKENIZER_DIRECTORY}/* /home/${USERNAME}/tokenizer/"
else
  echo "Copying tokenizer from local"
  gcloud compute tpus tpu-vm scp \
      --recurse \
      --zone="${ZONE}" \
      --project="${PROJECT_ID}" \
      --worker=all \
      ${LOCAL_TOKENIZER_DIRECTORY} ${TPU_NAME}:/home/${USERNAME}/tokenizer
fi
