#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t kitware/smqtk/iqr_playground_cpu -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
  nvidia-docker build -t kitware/smqtk/iqr_playground_nvidia \
    -f "${SCRIPT_DIR}/Dockerfile.nvidia" "${SCRIPT_DIR}"
fi
