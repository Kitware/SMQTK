#!/bin/bash
#
# Entry point for running the compute_many_descriptors.py script with standard
# IO.
#

set -e
now=$(date +%Y%m%d_%H%M%S)

function usage() {
  echo "$0 config filelist"
}

input_config="$1"
input_filelist="$2"

if [ ! -f "${input_config}" ]
then
  echo "ERROR No input config"
  usage
  exit 1
elif [ ! -f "${input_filelist}" ]
then
  echo "ERROR No input file list"
  usage
  exit 1
fi

scripts/compute_many_descriptors.py \
  -c "${input_config}" -f "${input_filelist}" \
  -p image_dump.${now}.computed_files.csv -v \
  2>&1 | tee image_dump.${now}.log.compute_many_descriptors.txt
