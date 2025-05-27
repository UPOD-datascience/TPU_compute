#!/bin/bash
set -o allexport
source ../.longformer.env
set +o allexport

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker=0
