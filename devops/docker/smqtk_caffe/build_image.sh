#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t kitware/smqtk/caffe:1.0-cpu \
             -f "${SCRIPT_DIR}/Dockerfile.cpu.df" \
             "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
    nvidia-docker build -t kitware/smqtk/caffe:1.0-gpu-cuda8.0-cudnn5 \
                        -f "${SCRIPT_DIR}/Dockerfile.gpu_cuda_cuDNNv5.df" \
                        "${SCRIPT_DIR}"
fi
