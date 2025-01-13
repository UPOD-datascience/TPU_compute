# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../../.env
set +o allexport

echo "Copying tokenizer from local"
gcloud compute tpus tpu-vm scp \
    --recurse \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --worker=all \
    ../../models/test.py ${TPU_NAME}:/home/${USERNAME}

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="PJRT_DEVICE=TPU python3 /home/${USERNAME}/test.py"
