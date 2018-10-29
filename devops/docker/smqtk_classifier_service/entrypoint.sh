#!/usr/bin/env bash
#
# Docker entrypoint script for the classifier service server.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_CONFIG_PATH="/configuration/server.json"

exec runApplication -vt \
    -a SmqtkClassifierService \
    -c "${SERVICE_CONFIG_PATH}"
