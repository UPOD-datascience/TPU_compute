set -o allexport
source ../.env
set +o allexport

gcloud compute tpus tpu-vm list --zone ${ZONE} --project ${PROJECT_ID}

gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}

gcloud compute tpus tpu-vm versions list --zone="${ZONE}"
