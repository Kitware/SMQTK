#!/usr/bin/env bash
#
# Execute full update stack
#
# Instructions:
#   - SMQTK setup script should be sourced before running this.
#   - Generate and modify configuration files for the following scripts:
#
#       <source>/bin/memex/list_ido_solr_images.py
#       <source>/bin/scripts/compute_many_descriptors.py
#       <source>/bin/scripts/compute_hash_codes.py
#       <source>/bin/scripts/compute_classifications.py
#
#   - Change/Update any settings values below upon first run as appropriate.
#   - Before each run, optionally update the "entries_after" setting with a
#     specific minimum "indexedAt" timestamp of images to process. If this is
#     empty, we will check if there are other directories in the ``run_dir``
#     and use their names (which are valid timestamps themselves) as the
#     minimum "indexedAt" times. If there are no other directories in the
#     ``run_dir``, we use "*" as the minimum timestamp (i.e. no minimum).
#
# This lets the process know to not consider images added
#     before that timestamp (they've already been processed). If this is the
#     first run, this should be set to "*".
#
#
# Configuration Notes:
#   - Data representation backends used should be ones that store data
#     persistently, otherwise this process will yield no net gain of
#     information when completed.
#
#
# Important result files:
#   - .../cmd.computed_files.csv
#       - Matches computed files with their SHA1 values, which is also used as
#         component UUIDs in the SMQTK ecosystem. This will contain the
#         absolute path of the file used locally, but due to the full relative
#         transfer during transfer, the full path on the remove server is
#         embedded inside this path. This file would be used to match ids to
#         SMQTK element UUIDs and vice versa.
#   - .../chc.hash2uuids.pickle
#       - This is the updated LSH algorithm hash mapping. This should be
#         swapped into place for new or live-updating systems using the LSH
#         neighbors index implementation.
#
set -e
set -o pipefail
set -u

#
# Settings
#
# update values below currently assigned teh ":: CHANGE ME ::" value.
#

# Script locations
script_lsi="scripts/list_ido_solr_images.py"
script_cmd="scripts/compute_many_descriptors.py"
script_chc="scripts/compute_hash_codes.py"
script_cc="scripts/compute_classifications.py"

# Configuration files
config_lsi="configs/config.list_solr_images.json"
config_cmd="configs/config.compute_many_descriptors.json"
config_chc="configs/config.compute_hash_codes.json"
config_cc="configs/config.compute_classifications.json"

# Base directory for intermediate and result files
run_dir="runs"

# Server where image files are located based on indexed paths in Solr instance
image_server="imagecat.dyndns.org"
# User to SSH into the above machine as
image_server_username=":: CHANGE ME ::"
# Global directory to sync imagery to
image_transfer_directory="images"

# We will collect/compute images that have been ingested after this time stamp.
# This may be be "*" to indicate no start bounds.
# - Format: "{Y}-{m}-{d}T{H}:{M}:{S}Z" or "*"
entries_after=  # none, use run directory logic

# Initial hash2uuids.pickle index to use as a base when computing new hash
# codes. The output index will include the content of this base. This may be
# empty to not include a base index.
base_hash2uuids="models/lsh.hash2uuid.pickle"


##################################
# DO NOT MODIFY BELOW THIS POINT #
##################################

function log() {
    echo "LOG :: $@"
}

# Timestamp in Solr time format
now="$(python -c "
import time
t=time.gmtime()
print '{Y:d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}Z'.format(
    Y=t.tm_year, m=t.tm_mon, d=t.tm_mday, H=t.tm_hour, M=t.tm_min, S=t.tm_sec
)")"

#
# Check configuration
#
if [ ! -f "${script_lsi}" ]; then
    echo "ERROR: Could not find '${script_lsi}' script"
    exit 1
elif [ ! -f "${script_cmd}" ]; then
    echo "ERROR: Could not find '${script_cmd}' script"
    exit 1
elif [ ! -f "${script_chc}" ]; then
    echo "ERROR: Could not find '${script_chc}' script"
    exit 1
elif [ ! -f "${script_cc}" ]; then
    echo "ERROR: Could not find '${script_cc}' script"
    exit 1
elif [ ! -f "${config_lsi}" ]; then
    echo "ERROR: Could not find '${config_lsi}' configuration file"
    exit 1
elif [ ! -f "${config_cmd}" ]; then
    echo "ERROR: Could not find '${config_cmd}' configuration file"
    exit 1
elif [ ! -f "${config_chc}" ]; then
    echo "ERROR: Could not find '${config_chc}' configuration file"
    exit 1
elif [ ! -f "${config_cc}" ]; then
    echo "ERROR: Could not find '${config_cc}' configuration file"
    exit 1
fi


# Working file paths
work_dir="${run_dir}/${now}"

# File marker of a complete run
complete_file=".complete"

# List of image files on remote server to compute over
remote_file_list="${work_dir}/file_list.remote.txt"
# List of image files locally to compute over after transfer
local_file_list="${work_dir}/file_list.local.txt"
# CSV mapping files descriptors were computed for with their SHA1/UUID values.
cmd_computed_files="${work_dir}/cmd.computed_files.csv"
# Listing of just UUID values (parsed from computed files csv).
uuids_list="${work_dir}/uuids_list.txt"
# Updated hash2uuids index pickle
hash2uuids_index="${work_dir}/hash2uuids.pickle"
# Output classifications columns header and data
classifications_header="${work_dir}/classifications.columns.csv"
classifications_data="${work_dir}/classifications.data.csv"

#
# Find last run timestamp if one wasn't manually provided and there is one
#
if [ -z "$entries_after" ]
then
    if [ -n "$(ls "$run_dir")" ]
    then
        entries_after="$(ls "$run_dir" | tail -n1)"
        # Only use this directory as a time stamp if it completed fully,
        # exiting if it did not
        if [ ! -f "$entries_after/$complete_file" ]
        then
            log "Previous run did not fully complete (missing completion marker)"
            exit 1
        fi
    else
        entries_after="*"
    fi
fi

mkdir -p "${work_dir}"

#
# Gather images from Solr instance
#
if [ ! -f "${remote_file_list}" ]; then
    log "Getting new remote image paths"
    log_remote_image_listing="${work_dir}/log.0.list_remote_files.txt"

    "${script_lsi}" -v -c "${config_lsi}" -p "${remote_file_list}" \
                    --after-time "${entries_after}" \
                    --before-time "${now}" \
                    2>&1 | tee "${log_remote_image_listing}"

    if [ ! -s "${remote_file_list}" ]; then
        log "No new image files since ${entries_after}"
        rm "${remote_file_list}"
        exit 0
    fi
fi

if [ ! -f "${local_file_list}" ]; then
    log "Gathering images from Solr"
    log_remote_local_rsync="${work_dir}/log.1.rsync.txt"

    rsync -PRvh --size-only --files-from="${remote_file_list}" \
          ${image_server_username}@${image_server}:/ \
          "${image_transfer_directory}" \
          2>&1 | tee "${log_remote_local_rsync}"

    python -c "
import os
base=os.path.abspath(os.path.expanduser('${image_transfer_directory}'))
with open('${remote_file_list}') as pth_f:
    for l in pth_f:
        pth = l.strip().lstrip('/')
        print os.path.join(base, pth)
    " >"${local_file_list}"

    if [ ! -s "${local_file_list}" ]; then
        log "Failed to generate local paths"
        rm "${local_file_list}"
        exit 1
    fi
fi

#
# Compute descriptors
#
if [ ! -f "${cmd_computed_files}" ]; then
    log "Computing descriptors on gathered images"
    log_cmd="${work_dir}/log.2.cmd.txt"

    "${script_cmd}" -v -c "${config_cmd}" -f "${local_file_list}" \
                    -p "${cmd_computed_files}" \
                    2>&1 | tee "${log_cmd}"
fi

if [ ! -f "${uuids_list}" ]; then
    cat "${cmd_computed_files}" | cut -d, -f2 | sort | uniq >"${uuids_list}"
fi

#
# Compute hash codes
#
if [ ! -f "${hash2uuids_index}" ]; then
    log "Computing hash codes"
    log_chc="${work_dir}/log.3.chc.txt"

    "${script_chc}" -v -c "${config_chc}" --uuids-list "${uuids_list}" \
                    --input-hash2uuids "${base_hash2uuids}" \
                    --output-hash2uuids "${hash2uuids_index}" \
                    2>&1 | tee "${log_chc}"
fi

log Re-symlink base with new index
rel_path="$(python -c "
import os
print os.path.relpath('${hash2uuids_index}', '$(dirname "${base_hash2uuids}")')
")"
ln -sf "${rel_path}" "${base_hash2uuids}"


#
# Compute classifications
#
if [ ! -s "${classifications_data}" ]; then
    log "Computing classifications"
    log_cc="${work_dir}/log.3.cc.txt"

    "${script_cc}" -v -c "${config_cc}" --uuids-list "${uuids_list}" \
                   --csv-header "${classifications_header}" \
                   --csv-data "${classifications_data}" \
                   2>&1 | tee "${log_cc}"
fi


# TODO: (?) Fit new ball tree


log Marking successful completeion
touch "${work_dir}/${complete_file}"
