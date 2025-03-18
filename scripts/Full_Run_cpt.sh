# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.cpt.env
set +o allexport

./delete.sh
./Step2a_startup_pod.sh
./Step2b_install_prequisites
./Step4_upload_scripts.sh
./Step5_tpu_cpt.sh

#gcloud compute tpus tpu-vm delete  \
#  --zone=${ZONE}
