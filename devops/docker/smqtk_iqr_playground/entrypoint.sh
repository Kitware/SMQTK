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
Usage: $0 [-b|--build [-t|--tile]]

Run SMQTK IQR GUI application over some images.
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

# Own psql/mongo/log directories for read/write
for DIR in {${PSQL_DATA_DIR},${MONGO_DATA_DIR}}
do
    echo "Owning dir: $DIR"
    sudo chown -R smqtk: "${DIR}"
done

# Create PSQL database if no exists
NEW_PSQL_DB=0
if [ -z "$(ls "${PSQL_DATA_DIR}" 2>/dev/null)" ]
then
    NEW_PSQL_DB=1
    pg_ctl -D "${PSQL_DATA_DIR}" init
    # Add socket to shared memory space
    echo "unix_socket_directories='/dev/shm'" >>"${PSQL_DATA_DIR}/postgresql.conf"
fi

# Start and background postgres and mongo
POSTGRES_PID="db.psql.pid"
postgres -D ${WORKING_DIR}/${PSQL_DATA_DIR} &>"${LOG_DIR}/db.psql.log" &
echo "$!" >"${POSTGRES_PID}"

MONGOD_PID="db.mongo.pid"
mongod --dbpath ${WORKING_DIR}/${MONGO_DATA_DIR} &>"${LOG_DIR}/db.mongo.log" &
echo "$!" >"${MONGOD_PID}"

# If a new database, add descriptors table to database
psql -h /dev/shm ${PSQL_NAME} ${PSQL_USER} -f "${CONFIG_DIR}/${PSQL_TABLE_INIT}"

################################################################################
# Run build if requested

if [ -n "${BUILD_MODELS}" ]
then
    LOG_GIT="${LOGS}/generate_image_transform.log"
    LOG_CMD="${LOGS}/compute_many_descriptors.log"
    LOG_ITQ="${LOGS}/train_itq.log"
    LOG_CHC="${LOGS}/compute_hash_codes.log"
    LOG_MBT="${LOGS}/make_balltree.log"

    # Create list of image files
    IMAGE_DIR_FILELIST="${IMAGE_DIR}.filelist.txt"
    find "${IMAGE_DIR}/" -type f >"${IMAGE_DIR_FILELIST}"

    if [ -n "${TILE_IMAGES}" ]
    then
        IMG_TILES_DIR="image_tiles"
        mkdir "${IMG_TILES_DIR}"
        cat "${IMAGE_DIR_FILELIST}" | parallel "
            generate_image_transform -vc "${CONFIG_DIR}/${SMQTK_GEN_IMG_TILES}" \
                -i \"{}\" -o \"${IMG_TILES_DIR}\"
        "
        # Use these tiles for new imagelist
        mv "${IMAGE_DIR_FILELIST}" >"${IMAGE_DIR_FILELIST}.ORIG"
        find "${IMG_TILES_DIR}" -type f >"${IMAGE_DIR_FILELIST}"
    fi

    # Tail build logs until they are done
    TAIL_PID="build_log_tail.pid"
    tail -F "${LOG_GIT}" "${LOG_CMD}" "${LOG_ITQ}" "${LOG_CHC}" "${LOG_MBT}" &
    echo "$!" >"${TAIL_PID}"

    # Compute descriptors
    compute_many_descriptors \
        -v -b ${SMQTK_CMD_BATCH_SIZE} --check-image -c "${SMQTK_CMD_CONFIG}" \
        -f "${IMAGE_DIR_FILELIST}" -p "${SMQTK_CMD_PROCESSED_CSV}" \
        &> "${LOGS}/compute_many_descriptors.log"

    # Train ITQ models
    train_itq -vc "${SMQTK_ITQ_TRAIN_CONFIG}" \
        &> "${LOGS}/train_itq.log"

    # Compute hash codes for descriptors
    compute_hash_codes \
        -vc "${SMQTK_HCODE_CONFIG}" \
        --output-hash2uuids "${SMQTK_HCODE_PICKLE}" \


    # Compute balltree hash index
    make_balltree "${SMQTK_HCODE_PICKLE}" ${SMQTK_ITQ_BIT_SIZE} \
        ${SMQTK_HCODE_BTREE_LEAFSIZE} ${SMQTK_HCODE_BTREE_RAND} \
        ${SMQTK_HCODE_BTREE_OUTPUT}

    # Stop log tail
    kill $(cat "${TAIL_PID}")
fi

################################################################################

SMQTK_IQR_PID="smqtk_iqr.pid"
runApplication \
    -a IqrSearchDispatcher \
    -vtc "${CONFIG_DIR}/${SMQTK_IQR_CONFIG}" \
    &>"${LOG_DIR}/runApp.IqrSearchDispatcher.log"
echo "$!" >"${SMQTK_IQR_PID}"


#
# Termination Wait
#
echo "Ctrl-C to exit or run 'docker stop <container>'"
wait \
    $(cat "${POSTGRES_PID}") \
    $(cat "${MONGOD_PID}") \
    $(cat "${SMQTK_IQR_PID}")


#
# Clean-up
#
echo "Stopping PostgreSQL"
kill $(cat "${POSTGRES_PID}")

echo "Stopping MongoDB"
kill $(cat "${MONGOD_PID}")

echo "Stopping SMQTK IqrSearchDispatcher"
kill $(cat "${SMQTK_IQR_PID}")

echo "exited $0"
