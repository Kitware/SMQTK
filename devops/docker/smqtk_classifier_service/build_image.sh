#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t kitware/smqtk/classifier_service:0.2-LRD-cpu \
             -f "${SCRIPT_DIR}/Dockerfile.cpu.df" \
             "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
    nvidia-docker build \
        -t kitware/smqtk/classifier_service:0.2-LRD-gpu-cuda8.0-cudnn6 \
        -f "$SCRIPT_DIR/Dockerfile.gpu-cuda8.0-cudnn6.df" \
        "${SCRIPT_DIR}"
fi
