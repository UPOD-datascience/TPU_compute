# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.llama.env
set +o allexport

./Step1b_create_disk.sh
./Step1c_upload_to_disk.sh
./Step2a_startup_pod.sh
./Step2b_install_prequisites.sh
./Step2c_mount_disk.sh
# ./Step4_upload_scripts.sh
# ./Step5_tpu_cpt.sh

#gcloud compute tpus tpu-vm delete  \
#  --zone=${ZONE}
