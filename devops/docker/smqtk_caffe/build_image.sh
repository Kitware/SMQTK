#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t kitware/smqtk/caffe_cpu \
             -f "${SCRIPT_DIR}/Dockerfile.cpu" \
             "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
    nvidia-docker build -t kitware/smqtk/caffe_nvidia \
                        -f "${SCRIPT_DIR}/Dockerfile.nvidia_cuDNNv5" \
                        "${SCRIPT_DIR}"
else
    echo "ERROR: Cannot build image, no nvidia-docker."
fi
