#!/usr/bin/env bash
set -e

function usage() {
  echo Usage: $0 FILEPATH INDEX_SIZE
}

filepath="$(readlink -f $1)"
n="$2"
sha1=$(sha1sum $filepath 2>/dev/null | cut -d' ' -f1)

if [ -z "${sha1}" ]
then
  echo "ERROR: Not a valid file path: ${filepath}"
  usage
  exit 1
fi
if [[ ! "$n" =~ ^[0-9]+$ ]]
then
  echo "ERROR: Not given a positive integer for the second parameter"
  usage
  exit 1
fi
if [ ! "$n" -gt 0 ]
then
  echo "ERROR: Index size must be >0 (given '$n')"
  usage
  exit 1
fi

# Usingi server address given default configuration.
if [ -n "$(curl http://127.0.0.1:5000/nn/n=${n}/file://${filepath} 2>/dev/null | grep "$sha1")" ]
then
  echo "File in index"
  exit 0
else
  echo "File NOT in index"
  exit 1
fi
