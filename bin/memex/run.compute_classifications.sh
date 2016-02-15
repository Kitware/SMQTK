#!/usr/bin/env bash
#
# Entry point for running the compute_classifications.py script with standard
# IO.
#
set -e
now=$(date +%Y%m%d_%H%M%S)

function usage() {
  echo "$0 config uuids_list"
}

input_config="$1"
input_uuids_list="$2"

if [ ! -f "${input_config}" ]
then
    echo "ERROR No input config"
    exit 1
elif [ ! -f "${input_uuids_list}" ]
then
    echo "ERROR No UUIDs list"
    exit 1
fi

output_csv_header="image_dump.${now}.classifications.header.csv"
output_csv_data="image_dump.${now}.classifications.data.csv"
output_log="image_dump.${now}.log.compute_classifications.txt"

scripts/compute_classifications.py \
    -v \
    -c "${input_config}" \
    --uuids-list "${input_uuids_list}" \
    --csv-header "${output_csv_header}" \
    --csv-data "${output_csv_data}" \
    2>&1 | tee "${output_log}"
