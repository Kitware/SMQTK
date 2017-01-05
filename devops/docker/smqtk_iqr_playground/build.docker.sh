#!/usr/bin/env bash
set -e
docker build -t kitware/smqtk/iqr_playground_cpu -f Dockerfile .

if [ -x "$(which nvidia-docker 2>/dev/null)" ]
then
  nvidia-docker build -t kitware/smqtk/iqr_playground_nvidia -f Dockerfile.nvidia .
fi
