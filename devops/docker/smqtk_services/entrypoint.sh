#!/usr/bin/env bash
set -e
trap "echo TRAPed signal" HUP INT QUIT KILL TERM
. /smqtk/install/setup_smqtk.sh

#
# 2 modes:
#   - Compute descriptors/models
#       - Compute descriptors and models based on service configuration file:
#         "/app/config.json"
#   - Host service
#       - host service based on configuration file: "/app/config.json"
#

#
# parameters!
#
NNSS_CONFIG_FP="/app/configs/nnss.config.json"
NNSS_LOG="/logs/smqtk.nnss.log"
NNSS_PID="/app/smqtk.nnss.pid"

IQR_CONFIG_FP="/app/configs/iqr.config.json"
IQR_LOG="/logs/smqtk.iqr.log"
IQR_PID="/app/smqtk.iqr.pid"

BUILD_MODELS=0

#
# Argument Parsing
#
while [[ $# -gt 0 ]]
do
key="$1"
shift  # past key

case ${key} in
    -b|--build-models)
    BUILD_MODELS=1
    ;;
    *)  # unknown option
    ;;
esac

done

# TODO: Check that database has the correct tables?

# Build descriptors+models if asked
if [ "${BUILD_MODELS}" -eq 1 ]
then
    echo "Computing Descriptors + Building models"
    /app/scripts/compute_models.sh 2>&1 | tee /logs/compute_models.log
fi

# TODO: Check that required models exist

#
# Run service(s)
#
echo "Starting NN Service"
runApplication.py \
    -a NearestNeighborServiceServer \
    -c "${NNSS_CONFIG_FP}" \
    -t --debug-smqtk --debug-server \
    &> "${NNSS_LOG}" &
echo "$!" >"${NNSS_PID}"

echo "Starting IQR service"
runApplication.py \
    -a IqrService \
    -c "${IQR_CONFIG_FP}" \
    -t --debug-smqtk --debug-server \
    &> "${IQR_LOG}" &
echo "$!" >"${IQR_PID}"

#
# Termination Wait
#
echo "Ctrl-C to exit or run 'docker stop <container>"
wait "$(cat "${NNSS_PID}")" "$(cat "${IQR_PID}")"

#
# Clean-up
#
echo "Stopping Nearest-Neighbor service"
kill $(cat "${NNSS_PID}")
rm "${NNSS_PID}"

echo "Stopping IQR service"
kill $(cat "${IQR_PID}")
rm "${IQR_PID}"

echo "exited $0"
