# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

./Step2a_startup_pod.sh
./Step2b_install_prequisites.sh
./Step3_upload_scripts.sh
./Step4_load_data.sh
./Step5_tpu_train_deberta.sh

#gcloud compute tpus tpu-vm delete  \
#  --zone=${ZONE}
