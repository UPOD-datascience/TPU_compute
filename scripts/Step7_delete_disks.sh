set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

gcloud alpha compute tpus tpu-vm detach-disk ${TPU_NAME} --zone=us-central1-f --disk=${EXT_DISK_NAME} || true
gcloud alpha compute tpus tpu-vm detach-disk ${TPU_NAME} --zone=us-central2-b --disk=${EXT_DISK_NAME} || true
gcloud alpha compute tpus tpu-vm detach-disk ${TPU_NAME} --zone=europe-west4-a --disk=${EXT_DISK_NAME} || true

gcloud compute disks delete ${EXT_DISK_NAME} --zone us-central1-f || true
gcloud compute disks delete ${EXT_DISK_NAME} --zone us-central2-b || true
gcloud compute disks delete ${EXT_DISK_NAME} --zone europe-west4-a || true
