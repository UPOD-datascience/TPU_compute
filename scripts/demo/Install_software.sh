# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../../.env
set +o allexport

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="
  pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--project=${PROJECT_ID} \
--worker=all --command="git clone -b r2.5 https://github.com/pytorch/xla.git"
