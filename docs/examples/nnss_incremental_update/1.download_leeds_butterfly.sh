#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

wget http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/files/leedsbutterfly_dataset_v1.0.zip
unzip leedsbutterfly_dataset_v1.0.zip
