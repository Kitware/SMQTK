#!/usr/bin/env bash
#
# SMQTK IQR entry-point script
#
# This script manages starting required services for running the
# IqrSearchDispatcher web application:
#   - PostgreSQL 9.4 using data directory ``/home/smqtk/data/db.psql``
#   - MongoDB using data directory ``/home/smqtk/data/db.mongo``
#   - SMQTK runApplication utility (runs IqrSearchDispatcher)
#
# In order for this container to be able to use the psql and mongo data
# directories, they will have to be chown'd here, thus they're permissions will
# change on the host system.
#
set -e

function usage() {
    echo "
Usage: $0 [-b|--build [-t|--tile]] [--rest]

Run SMQTK IQR GUI application and IQR REST service over some images.
Optionally, build required models for new imagery based on default or mounted
configs.

Options:

  -h | --help       Display this message

  -b | --build      Build model files for images mounted to
                    ``/home/smqtk/data/images``

  -t | --tile       Transform images found in the images directory according to
                    the provided ``generate_image_transform`` configuration JSON
                    file.
"
}

################################################################################
# Argument parsing

while [[ $# -gt 0 ]]
do
    key="$1"
    shift  # past key

    case ${key} in
        -h|--help)
        usage
        exit 1
        ;;
        -b|--build)
        BUILD_MODELS=1
        ;;
        -t|--tile)
        TILE_IMAGES=1
        ;;
        *)  # Anything else (wildcard)
        echo "Received unknown parameter: \"$key\""
        echo
        usage
        exit 1
        ;;
    esac

done

################################################################################
# Setup / configuration

WORKING_DIR="${HOME}/data"
ENTRYPOINT_CONFIG="${WORKING_DIR}/configs/entrypoint.conf"

if [ -f "${ENTRYPOINT_CONFIG}" ]
then
    echo "Setting entry-point variables from: '${ENTRYPOINT_CONFIG}'"
    source "${ENTRYPOINT_CONFIG}"
else
    echo "ERROR: Could not find entry-point config: '${ENTRYPOINT_CONFIG}'"
    exit 1
fi

pushd "${WORKING_DIR}"

################################################################################
# Start base services

# Own psql/mongo directories for read/write with special permissions
for DIR in {${PSQL_DATA_DIR},${MONGO_DATA_DIR}}
do
    echo "Claiming dir: $DIR"
    sudo chown -R smqtk: "${DIR}"
    # Database dirs needs limited permissions for security
    sudo chmod 700 "${DIR}"
done
sudo chown -R smqtk: "${LOG_DIR}"
sudo chmod +w "${LOG_DIR}"

# Create PSQL database if no exists
NEW_PSQL_DB=0
if [ -z "$(ls "${PSQL_DATA_DIR}" 2>/dev/null)" ]
then
    echo "Initializing PostgreSQL database"
    NEW_PSQL_DB=1
    pg_ctl -D "${PSQL_DATA_DIR}" init
    # Add socket to shared memory space
    echo "unix_socket_directories='/dev/shm'" >>"${PSQL_DATA_DIR}/postgresql.conf"
fi

# Start and background postgres and mongo
echo "Starting PSQL database server..."
POSTGRES_PID="db.psql.pid"
postgres -D ${WORKING_DIR}/${PSQL_DATA_DIR} &>"${LOG_DIR}/db.psql.log" &
echo "$!" >"${POSTGRES_PID}"
echo "Starting PSQL database server... Done"

echo "Starting MongoDB server..."
MONGOD_PID="db.mongo.pid"
mongod --dbpath ${WORKING_DIR}/${MONGO_DATA_DIR} &>"${LOG_DIR}/db.mongo.log" &
echo "$!" >"${MONGOD_PID}"
echo "Starting MongoDB server... Done"

# Wait for responsive database, upsert expected tables.
echo "Waiting for a responsive database..."
q=""
trigger="DatabaseNowResponsive"
while [ -z "$q" ]
do
  set +e
  q="$(psql -h "${PSQL_HOST}" ${PSQL_NAME} ${PSQL_USER} -c "\echo ${trigger}" 2>/dev/null | grep "${trigger}")"
  set -e
done
echo "Waiting for a responsive database... Done"
unset q trigger
echo "Creating (IF NOT EXISTS) required PostgreSQL tables..."
psql -h "${PSQL_HOST}" ${PSQL_NAME} ${PSQL_USER} -f "${CONFIG_DIR}/${PSQL_TABLE_INIT}"

################################################################################
# Run build if requested

if [ -n "${BUILD_MODELS}" ]
then
    STP_IMF="${LOG_DIR}/image_filelist_find.stamp"
    LOG_GIT="${LOG_DIR}/generate_image_transform.log"
    STP_GIT="${LOG_DIR}/generate_image_transform.stamp"
    LOG_CMD="${LOG_DIR}/compute_many_descriptors.log"
    STP_CMD="${LOG_DIR}/compute_many_descriptors.stamp"
    LOG_ITQ="${LOG_DIR}/train_itq.log"
    STP_ITQ="${LOG_DIR}/train_itq.stamp"
    LOG_CHC="${LOG_DIR}/compute_hash_codes.log"
    STP_CHC="${LOG_DIR}/compute_hash_codes.stamp"
    LOG_MBT="${LOG_DIR}/make_balltree.log"
    STP_MBT="${LOG_DIR}/make_balltree.stamp"

    echo "Owning model dir for writing"
    sudo chown -R smqtk: "${MODEL_DIR}"
    sudo chmod -R +rw "${MODEL_DIR}"

    # Create list of image files
    IMAGE_DIR_FILELIST="${IMAGE_DIR}.filelist.txt"
    if [ ! -e "${STP_IMF}" ]
    then
        find "${IMAGE_DIR}/" -type f >"${IMAGE_DIR_FILELIST}"
        touch "${STP_IMF}"
    fi

    # Tile discovered images if requested and hasn't happened yet.
    if [ -n "${TILE_IMAGES}" -a ! -e "${STP_GIT}" ]
    then
        echo "Generating tiles for images ($(wc -l "${IMAGE_DIR_FILELIST}" | cut -d' ' -f1) images)"
        IMG_TILES_DIR="image_tiles"
        mkdir -p "${IMG_TILES_DIR}"
        # Check if GNU parallel is available.
        if [ -n "$(which parallel 2>/dev/null)" ]
        then
            cat "${IMAGE_DIR_FILELIST}" | parallel "
                generate_image_transform -c \"${CONFIG_DIR}/${SMQTK_GEN_IMG_TILES}\" \
                    -i \"{}\" -o \"${IMG_TILES_DIR}\"
            "
        else
            cat "${IMAGE_DIR_FILELIST}" | \
                xargs -I '{}' generate_image_transform \
                    -c "${CONFIG_DIR}/${SMQTK_GEN_IMG_TILES}" -i '{}' \
                    -o "${IMG_TILES_DIR}"
        fi
        # Use these tiles for new imagelist
        mv "${IMAGE_DIR_FILELIST}" "${IMAGE_DIR_FILELIST}.ORIG"
        find "${IMG_TILES_DIR}" -type f >"${IMAGE_DIR_FILELIST}"
        touch "${STP_GIT}"
    fi

    # Tail build logs until they are done
    # - touch log files first to prevent tail warning about files not existing.
    TAIL_PID="build_log_tail.pid"
    touch "${LOG_GIT}" "${LOG_CMD}" "${LOG_ITQ}" "${LOG_CHC}" "${LOG_MBT}"
    tail -F "${LOG_GIT}" "${LOG_CMD}" "${LOG_ITQ}" "${LOG_CHC}" "${LOG_MBT}" &
    echo "$!" >"${TAIL_PID}"

    # Compute descriptors
    if [ ! -e "${STP_CMD}" ]
    then
        compute_many_descriptors \
            -v -b ${DESCRIPTOR_BATCH_SIZE} --check-image \
            -c "${CONFIG_DIR}/${SMQTK_CMD_CONFIG}" \
            -f "${IMAGE_DIR_FILELIST}" -p "${DESCRIPTOR_PROCESSED_CSV}" \
            &> "${LOG_CMD}"
        touch "${STP_CMD}"
    fi

    # Train ITQ models
    if [ ! -e "${STP_ITQ}" ]
    then
        train_itq -v -c "${CONFIG_DIR}/${SMQTK_ITQ_TRAIN_CONFIG}" \
            &> "${LOG_ITQ}"
        touch "${STP_ITQ}"
    fi

    # Compute hash codes for descriptors
    if [ ! -e "${STP_CHC}" ]
    then
        compute_hash_codes \
            -v -c "${CONFIG_DIR}/${SMQTK_CHC_CONFIG}" \
            &> "${LOG_CHC}"
        touch "${STP_CHC}"
    fi

    # Compute balltree hash index
    if [ ! -e "${STP_MBT}" ]
    then
        make_balltree -v -c "${CONFIG_DIR}/${SMQTK_MAKE_BALLTREE}" \
            &> "${LOG_MBT}"
        touch "${STP_MBT}"
    fi

    # Stop log tail
    kill $(cat "${TAIL_PID}")
fi

################################################################################

echo "Starting SMQTK IqrService..."
SMQTK_REST_IQR_PID="smqtk_rest_iqr.pid"
runApplication \
  -a IqrService \
  -vtc "${CONFIG_DIR}/${SMQTK_REST_IQR_CONFIG}" \
  &>"${LOG_DIR}/runApp.IqrService.log" &
echo "$!" >"${SMQTK_REST_IQR_PID}"
echo "Starting SMQTK IqrService... Done"

echo "Starting SMQTK IqrSearchDispatcher..."
SMQTK_GUI_IQR_PID="smqtk_iqr.pid"
runApplication \
  -a IqrSearchDispatcher \
  -vtc "${CONFIG_DIR}/${SMQTK_GUI_IQR_CONFIG}" \
  &>"${LOG_DIR}/runApp.IqrSearchDispatcher.log" &
echo "$!" >"${SMQTK_GUI_IQR_PID}"
echo "Starting SMQTK IqrSearchDispatcher... Done"



#
# Setup cleanup logic
#

#
# Wait on known processes
#
function process_pid_wait() {
  wait $(cat "${POSTGRES_PID}"       \
             "${MONGOD_PID}"         \
             "${SMQTK_REST_IQR_PID}" \
             "${SMQTK_GUI_IQR_PID}")
}

#
# Cleanup PID files
#
function process_pid_cleanup() {
  rm "${POSTGRES_PID}" \
     "${MONGOD_PID}" \
     "${SMQTK_REST_IQR_PID}" \
     "${SMQTK_GUI_IQR_PID}"
}

#
# Main cleanup function that stops running processes and cleans up after them
#
function process_cleanup() {
  signal="$1"

  # Because the input signal may have been propagated to sub-processes by the
  # OS and they may have terminated already.
  set +e

  echo "Stopping IQR REST Service"
  kill -${signal} $(cat "${SMQTK_GUI_IQR_PID}")

  echo "Stopping IQR GUI app"
  kill -${signal} $(cat "${SMQTK_REST_IQR_PID}")

  echo "Stopping MongoDB"
  kill -${signal} $(cat "${MONGOD_PID}")

  echo "Stopping PostgreSQL"
  kill -${signal} $(cat "${POSTGRES_PID}")

  set -e

  echo "Waiting on process completion..."
  process_pid_wait

  echo "Removing PID files..."
  rm "${POSTGRES_PID}" "${MONGOD_PID}"
  process_pid_cleanup
}

echo "Setting up cleanup trap"
trap "echo 'Terminating processes';   process_cleanup SIGTERM;" HUP INT TERM
trap "echo 'Force Killing processes'; process_cleanup SIGKILL;" QUIT KILL

#
# Termination Wait
#
echo "Ctrl-C to exit or run 'docker stop <container>'"
process_pid_wait

echo "exiting $0"
