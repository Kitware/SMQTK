#!/usr/bin/env sh
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
if [ -f ".coverage" ]
then
  echo "Removing previous coverage cache file"
  rm ".coverage"
fi
DEFAULT_ROOT="python/smqtk"
if [ "$#" -gt 0 ]
then
  nosetest_args="$@"
else
  nosetest_args="${DEFAULT_ROOT}"
fi
nosetests -v --with-doctest --with-coverage --cover-package=smqtk --exclude-dir-file=nose_exclude_dirs.txt ${nosetest_args}
