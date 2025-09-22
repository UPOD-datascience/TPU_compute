#!/bin/bash
#set -e

# Export variables from .env file
# set -o allexport
# source ../.llama.env
# set +o allexport

echo "Managing attachment and mounting of disk ${EXT_DISK_NAME} on TPU VM ${TPU_NAME}..."

# --- Disk Attachment ---

# Check if the specific disk is currently attached by looking for its full source path
echo "Checking if disk ${EXT_DISK_NAME} is attached to TPU ${TPU_NAME} and detaching if necessary..."
DISK_ATTACHED_SOURCE=$(gcloud compute tpus tpu-vm describe ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --format="value(disks.source)" 2>/dev/null | grep "${EXT_DISK_NAME}" || echo "")

if [[ ! -z "$DISK_ATTACHED_SOURCE" ]]; then
    echo "Disk ${EXT_DISK_NAME} is attached. Detaching..."
    # Attempt to detach, allowing failure if it's not actually attached anymore
    gcloud compute tpus tpu-vm detach-disk ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --disk=${EXT_DISK_NAME} || echo "Failed to detach ${EXT_DISK_NAME}, it may not be attached."

    echo "Waiting a moment after detachment..."
    sleep 10
else
    echo "Disk ${EXT_DISK_NAME} is not currently attached."
fi

# Now attach our disk
echo "Attaching disk ${EXT_DISK_NAME} to TPU ${TPU_NAME}..."

# Attempt to attach the disk, capturing stderr and suppressing stdout
ATTACH_ERROR=$(gcloud alpha compute tpus tpu-vm attach-disk ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --disk=${EXT_DISK_NAME} \
    --mode=${EXT_DISK_MODE} 2>&1 >/dev/null)

# Check the exit code of the attach command
ATTACH_EXIT_CODE=$?

# If the command failed (non-zero exit code)
if [ $ATTACH_EXIT_CODE -ne 0 ]; then
    # Check if the error message indicates the disk is already attached
    if echo "$ATTACH_ERROR" | grep -q "disk is already attached"; then
        echo "Warning: Disk ${EXT_DISK_NAME} was reported as already attached during attachment attempt. Continuing to mounting step."
        # Continue execution to the mounting step
    else
        # It's a different error, print the error and exit
        echo "Error during disk attachment:" >&2
        echo "$ATTACH_ERROR" >&2
        exit $ATTACH_EXIT_CODE # Exit with the original error code for other errors
    fi
else
    echo "Disk attached successfully."
fi

# Wait for the disk to be recognized by the OS
echo "Waiting for disk to be recognized by the OS after attachment..."
sleep 15

# --- Disk Mounting ---

# Create mount directory if it doesn't exist
echo "Creating mount point directory ${EXT_MOUNT_POINT} if it doesn't exist..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="sudo mkdir -p ${EXT_MOUNT_POINT}"

# Find the newly attached disk by UUID or LABEL and mount it
echo "Finding and mounting the newly attached disk (${EXT_DISK_NAME}) to ${EXT_MOUNT_POINT}..."

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="
    # Find the device path for the disk with the matching UUID or LABEL
    # We'll try LABEL first, then UUID as a fallback
    DISK_DEVICE=\"\"

    # Try finding by LABEL (assuming your disk is formatted with a label)
    DISK_DEVICE=\$(sudo blkid -L ${EXT_DISK_NAME} 2>/dev/null)

    # If not found by LABEL, try finding by UUID (if you know the UUID or can find it)
    # This part would require knowing the UUID, which might be tricky.
    # A more practical approach is to list all unmounted disks and try to match
    # based on size or filesystem type if you have a way to differentiate your disk.
    # For this script, let's stick to a more general approach: find any unmounted ext4 disk.

    if [ -z \"\$DISK_DEVICE\" ]; then
         # Find the disk by looking for an unmounted ext4 partition
         DISK_DEVICE=\$(sudo blkid -o device -t TYPE=\"ext4\" | while read DEV; do if ! mountpoint -q \$DEV; then echo \$DEV; break; fi; done)
    fi

    if [ -z \"\$DISK_DEVICE\" ]; then
        echo \"Error: Could not find the disk device for ${EXT_DISK_NAME} or any unmounted ext4 disk.\"
        exit 1
    fi

    echo \"Found target disk device: \$DISK_DEVICE. Attempting to mount...\"
    if sudo mount -o ro \$DISK_DEVICE ${EXT_MOUNT_POINT}; then
        echo \"Successfully mounted \$DISK_DEVICE to ${EXT_MOUNT_POINT}\"
    else
        echo \"Error: Failed to mount \$DISK_DEVICE to ${EXT_MOUNT_POINT}\"
        exit 1
    fi
    "

# Check if disk is mounted properly
echo "Verifying disk mount at ${EXT_MOUNT_POINT}..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="df -h | grep ${EXT_MOUNT_POINT} && ls -la ${EXT_MOUNT_POINT}/ || echo 'Warning: Disk not found at ${EXT_MOUNT_POINT} or directory is empty.'"

echo "Data disk management process completed for ${EXT_DISK_NAME}."
