#!/bin/bash

#
# Execute testing over the SMQTK Conductor package.
#
# Requires:
# - The SMQTK Conductor module is on the PYTHONPATH
# - python nose package installed
#

script_dir=$(cd "$(dirname "$0")"; pwd)
cover_dir=$script_dir/cover

. $script_dir/../setup_env.sh

echo
echo "================================================================="
echo "Testing SMQTK Backend python module."
echo
echo "Producing coverage report located"
echo "  @ $cover_dir"
echo
echo "=================================================================="
echo "NOTE: Safe to ignore Queue._feed traceback exceptions thrown,"
echo "      unless they were thrown from the main thread of execution"
echo "      (they usually aren't). They are happening on the separate"
echo "      Queue thread at shutdown. Tests are still passing, so"
echo "      everything should be fine... I don't know how to prevent"
echo "      them..."
echo "=================================================================="
echo
nosetests --with-doctest --doctest-options='+ELLIPSIS' --with-coverage --cover-erase --cover-html --cover-html-dir=$cover_dir --cover-package=SMQTK_Backend "$@"
