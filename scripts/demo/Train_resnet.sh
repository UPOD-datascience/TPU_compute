# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../../.env
set +o allexport

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="PJRT_DEVICE=TPU python3 ~/xla/test/test_train_mp_imagenet.py  \
  --fake_data \
  --model=resnet50  \
  --num_epochs=1 2>&1 | tee ~/logs.txt"

gcloud compute tpus tpu-vm scp \
    --recurse \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    ${TPU_NAME}:/home/${USERNAME}/logs.txt ./log.txt
