# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../../.env
set +o allexport

gcloud compute tpus tpu-vm create ${TPU_NAME} \
--zone=${ZONE} \
--project=${PROJECT_ID} \
--accelerator-type=${ACCELERATOR_TYPE} \
--version ${RUNTIME_VERSION}
