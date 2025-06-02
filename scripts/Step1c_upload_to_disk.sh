#!/bin/bash
set -e

# Export variables from .env file
# set -o allexport
# source ../.env
# set +o allexport

TEMP_VM_NAME="temp-disk-setup-vm"

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

# Check if the disk already exists
set +e
DISK_EXISTS=$(gcloud compute disks describe ${EXT_DISK_NAME} --zone=${ZONE} --project=${PROJECT_ID} --format="value(users)" 2>/dev/null)
set -e

if [[ ! -z "$DISK_EXISTS" ]]; then
    echo "Disk ${EXT_DISK_NAME} is currently attached to: $DISK_EXISTS"
    read -p "Do you want to detach the disk from these instances? (y/n): " DETACH_DISK

    if [[ "$DETACH_DISK" == "y" || "$DETACH_DISK" == "Y" ]]; then
        echo "Detaching disk ${EXT_DISK_NAME} from instances..."
        for INSTANCE in $DISK_EXISTS; do
            INSTANCE_NAME=$(echo $INSTANCE | awk -F/ '{print $NF}')
            echo "Detaching from $INSTANCE_NAME in zone $INSTANCE_ZONE..."
            gcloud compute instances detach-disk $INSTANCE_NAME --disk=${EXT_DISK_NAME} --zone=$ZONE --project=${PROJECT_ID}
        done
    else
        echo "Cannot proceed without detaching the disk. Exiting."
        exit 1
    fi
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
sleep 45  # Increased wait time to ensure disk is properly recognized

# Detect the actual device name of the attached disk
echo "Detecting actual device name of the attached disk..."
ACTUAL_DEVICE=$(gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="lsblk -d -o NAME,SERIAL | grep -i ${EXT_DISK_NAME} | awk '{print \$1}'" 2>/dev/null || echo "")

if [ -z "$ACTUAL_DEVICE" ]; then
    echo "Failed to detect device name automatically. Listing all block devices for reference:"
    gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="lsblk -d"
    echo "Falling back to specified device name: ${EXT_DISK_PART}"
    ACTUAL_DEVICE="${EXT_DISK_PART}"
else
    echo "Detected disk device name: /dev/${ACTUAL_DEVICE}"
fi

# Format disk if it's newly created (only if not already formatted)
echo "Checking if disk needs formatting..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="if [ -e /dev/${ACTUAL_DEVICE} ] && ! sudo blkid /dev/${ACTUAL_DEVICE}; then echo 'Formatting disk...'; sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/${ACTUAL_DEVICE}; else echo 'Disk already formatted or not found'; fi"

# Mount the disk
echo "Mounting disk to temporary VM..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="sudo mkdir -p ${EXT_MOUNT_POINT} && (sudo mount -o discard,defaults /dev/${ACTUAL_DEVICE} ${EXT_MOUNT_POINT} || (echo 'Disk already mounted or failed to mount. Listing current mounts:' && mount)) && sudo chmod 777 ${EXT_MOUNT_POINT}"

# Download the dataset from GCS
echo "Checking if dataset file already exists on disk..."
FILENAME=$(basename ${SHUFFLED_DATASET_GC})
FILE_EXISTS=$(gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="if [ -f ${EXT_MOUNT_POINT}/${FILENAME} ]; then echo 'exists'; else echo 'not_found'; fi")

if [ "$FILE_EXISTS" = "exists" ]; then
    echo "Dataset file ${FILENAME} already exists on disk. Skipping download."
else
    echo "Downloading dataset from ${SHUFFLED_DATASET_GC} to disk..."
    gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="gsutil -m cp ${SHUFFLED_DATASET_GC} ${EXT_MOUNT_POINT}/"
fi

# Show disk usage
echo "Disk usage after downloading:"
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="df -h ${EXT_MOUNT_POINT} && ls -la ${EXT_MOUNT_POINT}"

# Unmount the disk and clean up
echo "Unmounting disk and cleaning up..."
gcloud compute ssh ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --command="sudo umount ${EXT_MOUNT_POINT}"

# Delete the temporary VM
echo "Deleting temporary VM..."
gcloud compute instances delete ${TEMP_VM_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet

echo "Dataset successfully uploaded to disk ${EXT_DISK_NAME}. The disk is now ready to be attached to TPU in read-only mode."
