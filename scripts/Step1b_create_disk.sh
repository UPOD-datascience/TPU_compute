set -e

# Export variables from .env file
#set -o allexport
#source ../.env
#set +o allexport

if ! gcloud compute disks create ${EXT_DISK_NAME} \
    --size ${EXT_DISK_SIZE}  \
    --zone ${ZONE} \
    --type ${EXT_DISK_TYPE} 2>&1 | grep -q "already exists"; then
    :  # Command succeeded, do nothing
else
    echo "disk already exists, continuing"
fi
    #--access-mode ${EXT_DISK_MODE}
