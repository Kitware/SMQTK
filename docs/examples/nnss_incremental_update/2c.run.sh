#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

../../../bin/scripts/compute_hash_codes.py -v -c 2c.config.compute_hash_codes.json
