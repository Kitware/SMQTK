#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
    nvidia-docker build -t kitware/smqtk/caffe_nvidia \
                        "${SCRIPT_DIR}"
else
    echo "ERROR: Cannot build image, no nvidia-docker."
fi
