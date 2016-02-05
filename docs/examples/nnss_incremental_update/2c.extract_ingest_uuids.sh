#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

cat 2a.completed_files.csv | cut -d',' -f2 >2c.uuids_for_processing.txt
