# Exit immediately if a command exits with a non-zero status
# set -e

# # Export variables from .env file
# set -o allexport
# source ../.env
# set +o allexport

gcloud compute tpus tpu-vm list --zone="${ZONE}" --project="${PROJECT_ID}" | grep -q "${TPU_NAME}"
grep_exit_code=$?
set -e

if [ $grep_exit_code -eq 0 ]; then
    echo "TPU ${TPU_NAME} exists, deleting."
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT_ID}" \
        --quiet
else
    echo "No TPU ${TPU_NAME} found."
fi
