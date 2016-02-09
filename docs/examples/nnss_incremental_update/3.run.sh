#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Compute descriptors for new files, outputing a file that matches input
# files to thair SHA1 checksum values (their UUIDs)
../../../bin/scripts/compute_many_descriptors.py \
  -d \
  -c 2a.config.compute_many_descriptors.json \
  -f 3.ingest_files_2.txt \
  --completed-files 3.completed_files.csv

# Extract UUIDs of files/descriptors just generated
cat 3.completed_files.csv | cut -d, -f2 > 3.uuids_for_processing.txt

# Compute hash codes for descriptors just generated, updating the target
# hash2uuids model file.
../../../bin/scripts/compute_hash_codes.py -v -c 3.config.compute_hash_codes.json
