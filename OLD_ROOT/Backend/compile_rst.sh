#!/bin/env bash

#
# Find RST files underneath this script and compile them to HTML
#

# required rst2html.py, error out if not found in path
which rst2html.py >/dev/null 2>&1
if [ ! $? -eq 0 ]
then
  echo "ERROR: no 'rst2html.py' found in the current PATH."
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo
for F in $(find "$script_dir" -name "*.rst")
do
  echo "$F"
  echo "└─ ${F%.rst}.html"
  rst2html.py $F ${F%.rst}.html
  echo
done

