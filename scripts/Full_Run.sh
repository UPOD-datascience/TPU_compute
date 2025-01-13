# Exit immediately if a command exits with a non-zero status
set -e

# Export variables from .env file
set -o allexport
source ../.env
set +o allexport

./Step2a_startup_pod.sh
./Step2b_install_prequisites_v1.sh
./Step3_load_data.sh
#./Step4_train_tokenizer.sh
./Step5_tpu_train_deberta_v1.sh

gcloud compute tpus tpu-vm delete  \
  --zone=${ZONE}
