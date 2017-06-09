#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build \
  -t kitware/smqtk/iqr_playground_cpu \
  -f "${SCRIPT_DIR}/Dockerfile.cpu.df" \
  "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
  nvidia-docker build \
    -t kitware/smqtk/iqr_playground_nvidia \
    -f "${SCRIPT_DIR}/Dockerfile.gpu.df" \
    "${SCRIPT_DIR}"
fi
