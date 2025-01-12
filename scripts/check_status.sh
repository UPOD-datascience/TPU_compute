set -o allexport
source ../.env
set +o allexport

gcloud compute tpus tpu-vm list --zone ${ZONE} --project ${PROJECT_ID}

gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}

gcloud compute tpus tpu-vm versions list --zone="${ZONE}"

echo "Checking TPU status..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --command "python -c 'import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())'"

echo "Starting test..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="
sudo pkill -f python3 &&
python3 /home/${USERNAME}/models/test.py"
