#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker pull ubuntu:16.04
docker build -t kitware/smqtk/caffe:1.0-cpu \
             -f "${SCRIPT_DIR}/Dockerfile.cpu.df" \
             "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
    docker pull nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
    nvidia-docker build -t kitware/smqtk/caffe:1.0-gpu-cuda8.0-cudnn6 \
                        -f "${SCRIPT_DIR}/Dockerfile.gpu_cuda_cuDNNv6.df" \
                        "${SCRIPT_DIR}"
fi
