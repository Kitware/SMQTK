#!/bin/bash
#
# Simple script for starting the SMQTK IQR container over a directory of
# images. The ``-t`` option may be optionally provided to tile input imagery
# into 64x64 tiles (default). We drop into watching the processing status after
# starting the container.
#
# If the container is already running, we just start watching the container's status.
#
set -e

CONTAINER_NAME="smqtk_iqr_gpu"

if [ -z "$( docker ps -a | grep "${CONTAINER_NAME}" 2>/dev/null )" ]
then
  IMAGE_DIR="$1"
  shift
  nvidia-docker run -d -p 5000:5000 -v "${IMAGE_DIR}":/home/smqtk/data/images --name "${CONTAINER_NAME}" kitware/smqtk/iqr_playground_nvidia:pascal_patched -b "$@"
fi

watch -n1 "
nvidia-smi
echo
docker exec ${CONTAINER_NAME} bash -c 'echo \"Image tiles generated: \$(ls data/image_tiles | wc -l)\"'
echo
docker exec ${CONTAINER_NAME} tail data/logs/compute_many_descriptors.log data/logs/train_itq.log data/logs/compute_hash_codes.log data/logs/runApp.IqrSearchDispatcher.log
"
