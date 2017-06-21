#!/usr/bin/env bash
set -e
#
# "Big-red-button" script to start up a SMQTK Nearest-Neighbors service around
# a directory of images.
#
# Usage:
#   smqtk_services.run_images.sh -i IMAGE_DIR_PATH
#

IMAGE_DIR=""

#
# Argument Parsing
#
while [[ $# -gt 0 ]]
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
    (
        echo -n "ERROR: No image directory provided."
        echo -n " Please provide -i|--images option."
        echo
    ) >&2

    exit 1
elif [ -z "$(ls "${IMAGE_DIR}")" ]
then
    echo "ERROR: Nothing in provided image directory: ${IMAGE_DIR}" >&2
    exit 2
fi

export IMAGE_DIR
mkdir -p smqtk_logs
docker-compose up -d postgres

# Wait until PSQL instance is up and running by poking psql
trigger="DatabaseNowResponsive"
poll_command='docker-compose exec postgres psql postgres postgres'
poll_command="$poll_command -c \"\\\\echo $trigger\" 2>/dev/null"
poll_command="$poll_command | grep -q \"$trigger\""

echo "Waiting for a responsive database"

(
    set +e
    eval "$poll_command"
    while (("$?")) ; do eval "$poll_command" ; done
)

# Create new tables in DB, pulling init scripts from SMQTK container
#
# workaround https://github.com/docker/compose/issues/3352
# When above issue is resolved, replace
#
#     docker exec -i $(docker-compose ps -q postgres) ...
#
# with
#
#     docker-compose exec postgres ...
#
echo "Creating required tables"
docker exec -i $(docker-compose ps -q postgres) \
    psql postgres postgres 1>/dev/null << EOSQL
$(docker run --rm --entrypoint cat kitware/smqtk \
  /smqtk/install/etc/smqtk/postgres/descriptor_element/example_table_init.sql \
  /smqtk/install/etc/smqtk/postgres/descriptor_index/example_table_init.sql)
EOSQL

#
# Build models and start services
#
# Given the "-b" argument, which tells the container to build models using
# default configuration.
#
echo "Starting SMQTK Services docker"
docker-compose up -d smqtk wrapper
