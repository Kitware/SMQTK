#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
if [ -f ".coverage" ]
then
  echo "Removing previous coverage cache file"
  rm .coverage*
fi
DEFAULT_ROOT="python/smqtk"
if [ "$#" -gt 0 ]
then
  test_paths="$@"
else
  test_paths="${DEFAULT_ROOT}"
fi

# -l :: Show local in trace-backs.
# -v :: Increased verbosity.
pytest -lv \
  --cov=smqtk \
  ${test_paths}
