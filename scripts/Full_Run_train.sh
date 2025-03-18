# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

./Step2a_startup_pod.sh
./Step2b_install_prequisites.sh
./Step4_upload_scripts.sh
./Step5_tpu_train.sh

#gcloud compute tpus tpu-vm delete  \
#  --zone=${ZONE}
