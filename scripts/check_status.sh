set -o allexport
source ../.env
set +o allexport

echo "Starting test..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="
python3 /home/${USERNAME}/models/test.py"

gcloud compute tpus tpu-vm list --zone ${ZONE} --project ${PROJECT_ID}

gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}

gcloud compute tpus tpu-vm versions list --zone="${ZONE}"
