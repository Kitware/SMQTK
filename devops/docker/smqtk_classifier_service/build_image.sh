#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"

docker build -t kitware/smqtk/classifier_service:cpu \
             -f "${SCRIPT_DIR}/Dockerfile.cpu.df" \
             "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
    nvidia-docker build -t kitware/smqtk/classifier_service:gpu-cuda8.0-cudnn5 \
                        -f "$SCRIPT_DIR/Dockerfile.gpu-cuda8.0-cudnn5.df" \
                        "${SCRIPT_DIR}"
fi
