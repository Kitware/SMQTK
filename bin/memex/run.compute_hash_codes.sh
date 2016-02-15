#!/usr/bin/env bash
#
# Entry point for running the compute_hash_codes.py script with standard IO.
#
set -e

function usage() {
  echo "$0 config computed_files"
}

input_config="$1"
input_computed_files="$2"
now=$(date +%Y%m%d_%H%M%S)

if [ ! -f "${input_config}" ]
then
  echo "ERROR No input config given"
  exit 1
elif [ ! -f "${input_computed_files}" ]
then
  echo "ERROR No computed files CSV given"
  exit 1
fi

# Translate computed files CSV into just a UUIDs list
uuids_list="image_dump.${now}.uuids_list.txt"
cat "${input_computed_files}" | cut -d, -f2 | sort | uniq >"${uuids_list}"

# Compute hash codes for UUIDs
models_dir="models"
base_h2u="lsh.hash2uuid.pickle"
output_h2u="lsh.hash2uuid.${now}.pickle"
output_log="image_dump.${now}.log.compute_hash_codes.txt"
scripts/compute_hash_codes.py -v \
  -c "${input_config}" \
  --uuids-list "${uuids_list}" \
  --input-hash2uuids "${models_dir}/${base_h2u}" \
  --output-hash2uuids "${models_dir}/${output_h2u}" \
  2>&1 | tee "${output_log}"
if [ "$?" -gt 0 ]
then
  echo "ERROR: Failed hash code generation step"
fi

echo "Swaping hash2uuid model links"
ln -sf "${output_h2u}" "${models_dir}/${base_h2u}"
