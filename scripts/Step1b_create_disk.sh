set -e

# Export variables from .env file
#set -o allexport
#source ../.env
#set +o allexport

gcloud compute disks create ${EXT_DISK_NAME} \
    --size ${EXT_DISK_SIZE}  \
    --zone ${ZONE} \
    --type ${EXT_DISK_TYPE}
    #--access-mode ${EXT_DISK_MODE}
