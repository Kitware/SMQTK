#!/usr/bin/env bash
set -e
#
# "Big-red-button" script to start up a SMQTK Nearest-Neighbors service around
# a directory of images.
#

PREFIX=smqtk
DOCKER_POSTGRES="${PREFIX}-postgres"
DOCKER_SMQTK="${PREFIX}-services"
IMAGE_DIR=""

#
# Argument Parsing
#
while [[ $# > 0 ]]
do
key="$1"
shift  # past key
case ${key} in
    -i|--images)
    IMAGE_DIR="$1"
    shift  # past value
    ;;
    *)  # unknown option
    ;;
esac
done

# Argument error checking
if [ -z "${IMAGE_DIR}" ]
then
    echo "ERROR: No image directory provided. Please provide -i|--images option."
    exit 1
elif [ -z "$(ls "${IMAGE_DIR}")" ]
then
    echo "ERROR: Nothing in provided image directory: ${IMAGE_DIR}"
    exit 2
fi

#
# Start and Initialize PostgreSQL container
#
echo "Starting up PostgreSQL docker"
docker run -d --name "${DOCKER_POSTGRES}" postgres

# Wait until PSQL instance is up and running by poking psql
echo "Waiting for a responsive database"
q=""
trigger="DatabaseNowResponsive"
while [ -z "$q" ]
do
    set +e
    q="$(docker exec ${DOCKER_POSTGRES} psql postgres postgres -c "\echo ${trigger}" 2>/dev/null | grep "${trigger}")"
    set -e
done
unset q trigger

# Create new tables in DB, pulling init scripts from SMQTK container
echo "Creating required tables"
docker exec -i ${DOCKER_POSTGRES} psql postgres postgres 1>/dev/null <<-EOSQL
    $(docker run --rm --entrypoint bash ${DOCKER_SMQTK} \
        -c "cat \
            /smqtk/install/etc/smqtk/postgres/descriptor_element/example_table_init.sql \
            /smqtk/install/etc/smqtk/postgres/descriptor_index/example_table_init.sql")
EOSQL

#
# Build models and start services
#
echo "Starting SMQTK Services docker"
mkdir -p smqtk_logs
docker run -d --name ${DOCKER_SMQTK} \
    --link ${DOCKER_POSTGRES}:postgres \
    -v "${IMAGE_DIR}":/data \
    -v $PWD/smqtk_logs:/logs \
    -p 12345:12345 \
    -p 12346:12346 \
    smqtk-services -b

# Tail the expected logs
tail -f smqtk_logs/compute_models.log smqtk_logs/smqtk.nnss.log
