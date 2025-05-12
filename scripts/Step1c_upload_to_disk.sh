#!/bin/bash
set -e

# Export variables from .env file
#set -o allexport
#source ../.cpt.env
#set +o allexport

TEMP_VM_NAME="temp-disk-setup-vm"
DISK_MOUNT_DIR="/mnt/data"

echo "Creating temporary VM to prepare data disk..."

# Check if the temporary VM already exists
set +e
gcloud compute instances describe ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} &>/dev/null
VM_EXISTS=$?
set -e

# If VM exists, delete it
if [ $VM_EXISTS -eq 0 ]; then
    echo "Temporary VM already exists. Deleting it..."
    gcloud compute instances delete ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet
fi

# Create a temporary VM to mount the disk
echo "Creating temporary VM: ${TEMP_VM_NAME}..."
gcloud compute instances create ${TEMP_VM_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --machine-type=n2-standard-8 \
    --disk="name=${EXT_DISK_NAME},device-name=${EXT_DISK_NAME},mode=rw" \
    --scopes=cloud-platform

echo "Waiting for VM to initialize..."
sleep 30

# Format disk if it's newly created (only if not already formatted)
echo "Checking if disk needs formatting..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="if ! sudo blkid /dev/sdb; then echo 'Formatting disk...'; sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb; fi"

# Mount the disk
echo "Mounting disk to temporary VM..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="sudo mkdir -p ${DISK_MOUNT_DIR} && sudo mount -o discard,defaults /dev/sdb ${DISK_MOUNT_DIR} && sudo chmod 777 ${DISK_MOUNT_DIR}"

# Download the dataset from GCS
echo "Downloading dataset from ${SHUFFLED_DATASET_GC} to disk..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="gsutil -m cp ${SHUFFLED_DATASET_GC} ${DISK_MOUNT_DIR}/"

# Show disk usage
echo "Disk usage after downloading:"
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="df -h ${DISK_MOUNT_DIR} && ls -la ${DISK_MOUNT_DIR}"

# Unmount the disk and clean up
echo "Unmounting disk and cleaning up..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="sudo umount ${DISK_MOUNT_DIR}"

# Delete the temporary VM
echo "Deleting temporary VM..."
gcloud compute instances delete ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet

echo "Dataset successfully uploaded to disk ${EXT_DISK_NAME}. The disk is now ready to be attached to TPU in read-only mode."
