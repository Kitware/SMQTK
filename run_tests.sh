#!/usr/bin/env bash
set -e

# Make sure we are in the appropriate directory.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

# Remove any lingering coverage files from previous runs.
if [ -f ".coverage" ]
then
  echo "Removing previous coverage cache file"
  rm .coverage*
fi

# Clean up any *.py[co] files that may include no longer relevant code and/or
# tests.
PYCO_FILES="$(find ${script_dir} -name "*.py[co]")"
if [ -n "${PYCO_FILES}" ]
then
    PYCO_FILES_COUNT="$(echo ${PYCO_FILES} | sed -re "s| |\n|g" | wc -l)"
    echo "Removing ${PYCO_FILES_COUNT} old \"*.py[co]\" files"
    rm ${PYCO_FILES}
fi

pytest "$@"
