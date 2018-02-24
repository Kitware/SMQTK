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
#     specific minimum "indexedAt" timestamp of images to process (time the
#     image was crawled). If this is empty, we will check if there are other
#     directories in the ``run_dir`` and use their names (which are valid
#     timestamps themselves) as the minimum "indexedAt" times. If there are no
#     other directories in the ``run_dir``, we use "*" as the minimum timestamp
#     (i.e. no minimum).
#
# This lets the process know to not consider images added
#     before that timestamp (they've already been processed). If this is the
#     first run, this should be set to "*".
#
#
# Confirming that everything was run that was able
# ------------------------------------------------
# It is possible (even probable) that the number of remote files fetched
# from the Solr index (listed in the ``file_list.local.txt``) will be greater
# than the number of descriptors actually generated and inserted into the
# configured DescriptorIndex. This is due to multiple points at which an image
# may be invalidated or considered unfit for processing. The following are
# things to look for in the ``log.2.cmd.txt`` file to trace why fewer
# descriptors were possibly generated than input files:
#
#     - Image could be skipped due to being a file type the descriptor
#       generator algorithm cannot handle.
#         - Search for the text: "Skipping file"
#     - The image file could not be loaded for some reason
#         - Search for the text: "Failed to convert"
#     - Multiple images in the input set were duplicates of each other.
#         - This can be seen by either looking at:
#             - the number of unique UUIDs (SHA1 checksums) vs input files in
#               the ``cmd.computed_files.csv`` output file.
#             - the difference in the counts tracked in the log file (search
#               for "Processed \d+ so far (\d+ total data elements input)").
#
# The total number of remote files transferred minus the counts for bullets one
# and two above should equal the number of files for which descriptors were
# computed (lines in ``cmd.computed_files.csv``). The number of unique
# descriptors, however, should be equal to the number of unique UUIDs listed in
# ``cmd.computed_files.csv`` (extracted into the ``uuids_list.txt`` file).
#
#
# Configuration Notes
# -------------------
# - Data representation backends used should be ones that store data
#   persistently, otherwise this process will yield no net gain of
#   information when completed.
#
#
# Important result files
# ----------------------
# - .../cmd.computed_files.csv
#     - Matches computed files with their SHA1 values, which is also used as
#       component UUIDs in the SMQTK ecosystem. This will contain the
#       absolute path of the file used locally, but due to the full relative
#       transfer during transfer, the full path on the remove server is
#       embedded inside this path. This file would be used to match ids to
#       SMQTK element UUIDs and vice versa.
# - .../chc.hash2uuids.pickle
#     - This is the updated LSH algorithm hash mapping. This should be
#       swapped into place for new or live-updating systems using the LSH
#       neighbors index implementation.
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
script_gci="scripts/get_cdr_images.py"
script_lsi="scripts/list_ido_solr_images.py"
script_cmd="scripts/compute_many_descriptors.py"
script_chc="scripts/compute_hash_codes.py"
script_cc="scripts/compute_classifications.py"

# Configuration files
config_gci="configs/config.get_cdr_images.json"
config_lsi="configs/config.list_ido_solr_images.json"
config_cmd="configs/config.compute_many_descriptors.json"
config_chc="configs/config.compute_hash_codes.json"
config_cc="configs/config.compute_classifications.json"

# Base directory for intermediate and result files
run_dir="runs"

# Batch size for descriptor computation
cmd_batch_size=1024

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

# We will either get images from the configured CDR location or the paired Solr
# instance and file host.
# Valid values: "cdr" | "solr"
transfer_method="cdr"


##################################
# DO NOT MODIFY BELOW THIS POINT #
##################################

function log() {
    echo "LOG :: $@"

    if [ -f "${work_log}" ]
    then
        echo "LOG :: $@" >>"${work_log}"
    fi
}

function error() {
    echo "ERROR :: $@"

    if [ -f "${work_log}" ]
    then
        echo "ERROR :: $@" >>"${work_log}"
    fi
}

# Timestamp in Solr time format
now="$(python -c "
import time
t=time.gmtime()
print('{Y:d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}Z'.format(
    Y=t.tm_year, m=t.tm_mon, d=t.tm_mday, H=t.tm_hour, M=t.tm_min, S=t.tm_sec
))")"
set +u
if [ -n "$1" ]
then
    log "Manually specified time entry / run-dir: '$1'"
    now="$1"
    # Checking timestamp format validity
    set +e
    python -c """
import re
m = re.match('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '$1')
if m is None:
    exit(1)
exit(0)
"""
    if [ "$?" -eq 1 ]
    then
        error "Manual entry of incorrect timestamp format."
        exit 1
    fi
    set -e
fi
set -u


#
# Check configuration
#
if [ "${transfer_method}" = "cdr" -a ! -f "${script_gci}" ]; then
    error "Could not find '${script_gci}' script"
    exit 1
elif [ "${transfer_method}" = "solr" -a ! -f "${script_lsi}" ]; then
    error "Count not find '${script_lsi}' script"
    exit 1
elif [ ! -f "${script_cmd}" ]; then
    error "Could not find '${script_cmd}' script"
    exit 1
elif [ ! -f "${script_chc}" ]; then
    error "Could not find '${script_chc}' script"
    exit 1
elif [ ! -f "${script_cc}" ]; then
    error "Could not find '${script_cc}' script"
    exit 1
elif [ "${transfer_method}" = "cdr" -a ! -f "${config_gci}" ]; then
    error "Could not find '${config_gci}' configuration file"
    exit 1
elif [ "${transfer_method}" = "solr" -a ! -f "${config_lsi}" ]; then
    error "Could not find '${config_lsi}' configuration file"
    exit 1
elif [ ! -f "${config_cmd}" ]; then
    error "Could not find '${config_cmd}' configuration file"
    exit 1
elif [ ! -f "${config_chc}" ]; then
    error "Could not find '${config_chc}' configuration file"
    exit 1
elif [ ! -f "${config_cc}" ]; then
    error "Could not find '${config_cc}' configuration file"
    exit 1
fi


# Working file paths
work_dir="${run_dir}/${now}"
work_log="${work_dir}/log.update.txt"

# File marker of a complete run
complete_file=".complete"

# List of image files on remote server to compute over (Solr transfer ONLY)
remote_file_list="${work_dir}/file_list.remote.txt"
# CDR fetch records (CDR transfer ONLY)
cdr_fetch_record="${work_dir}/cdr_fetch_record.csv"
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
log "run_dir = '$run_dir'"
log "now = '${now}'"

if [ -z "${entries_after}" ]
then
    # if grep doesn't match anything, it returns a non-zero code
    set +e
    last_run="$(ls "${run_dir}" | grep -v "${now}" | tail -n1)"
    set -e
    log "last run: '${last_run}'"

    if [ -n "${last_run}" ]
    then
        # Only use this directory as a time stamp if it completed fully,
        # exiting if it did not
        if [ ! -f "${run_dir}/${last_run}/${complete_file}" ]
        then
            error "Previous run did not fully complete (missing completion marker)"
            error "(missing '${run_dir}/${last_run}/${complete_file}')"
            exit 1
        fi
        entries_after="${last_run}"
    else
        log "No previous runs"
        entries_after="*"
    fi
fi
log "Fetching Entries After: '${entries_after}'"

mkdir -p "${work_dir}"
touch "${work_log}"  # Start log recording


#
# Get images from source
#
if [ ! -f "${local_file_list}" ]
then
    if [ "${transfer_method}" = "cdr" ]
    then
        #
        # CDR source
        #
        log "Fetching new image content from CDR"

        log_gci="${work_dir}/log.1.gci.txt"

        after_time_opt=""
        if [ "${entries_after}" != "*" ]
        then
            after_time_opt="--crawled-after ${entries_after}"
        fi

        "${script_gci}" -v -c "${config_gci}" -d "${image_transfer_directory}" \
                        -l "${cdr_fetch_record}" ${after_time_opt} \
                        2>&1 | tee "${log_gci}"

        # Transform record file into local file image list
        cat "${cdr_fetch_record}" | cut -d, -f2 >"${local_file_list}"

    elif [ "${transfer_method}" = "solr" ]
    then
        #
        # Solr / Server source
        #
        log "Gathering images from Solr"

        if [ ! -f "${remote_file_list}" ]
        then
            log "Getting new remote image paths after_time='${entries_after}' before_time='${now}'"
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

        log_remote_local_rsync="${work_dir}/log.1.rsync.txt"
        # Some files might not transfer or paths in index might have been incorrect
        # (its happened before). Will check for actually transferred files right
        # after the rsync.
        set +e
        rsync -PRvh --size-only --files-from="${remote_file_list}" \
              ${image_server_username}@${image_server}:/ \
              "${image_transfer_directory}" \
              2>&1 | tee "${log_remote_local_rsync}"
        set -e

        log "Finding transferred files..."
        python -c "
import os
base=os.path.abspath(os.path.expanduser('${image_transfer_directory}'))
with open('${remote_file_list}') as pth_f:
    for l in pth_f:
        pth = l.strip().lstrip('/')
        local = os.path.join(base, pth)
        if os.path.isfile(local):
            print(local)
        " >"${local_file_list}"

        if [ ! -s "${local_file_list}" ]; then
            log "Failed to transfer any remote files locally"
            rm "${local_file_list}"
            exit 1
        fi
    else
        error "Invalid transfer method setting: '${transfer_method}'"
        exit 1
    fi
fi

if [ "$(cat "${local_file_list}" | wc -l)" -eq 0 ]
then
    error "No local files downloaded"
    exit 1
fi


#
# Compute descriptors
#
if [ ! -f "${cmd_computed_files}" ]; then
    log "Computing descriptors on gathered images"
    log_cmd="${work_dir}/log.2.cmd.txt"

    "${script_cmd}" -v -c "${config_cmd}" -f "${local_file_list}" \
                    -p "${cmd_computed_files}" -b "${cmd_batch_size}" \
                    2>&1 | tee "${log_cmd}"

    # Validate the number of descriptors generated (output UUIDs)
    log "Validating computed descriptor counts"
    num_input_files=$(cat "${local_file_list}" | wc -l)
    num_bad_ct=$(grep "Skipping file" "${log_cmd}" | wc -l)
    num_bad_file=$(grep "Failed to convert" "${log_cmd}" | wc -l)
    expected_processed=$(echo ${num_input_files} - ${num_bad_ct} - ${num_bad_file} | bc)
    true_processed=$(cat "${cmd_computed_files}" | wc -l)
    if [ "${expected_processed}" -ne "${true_processed}" ]
    then
        error "Could not account for processed output counts "\
              "(${expected_processed} != ${true_processed})"
        exit 1
    fi
fi

if [ ! -f "${uuids_list}" ]; then
    log "Extracting UUIDs from computed descriptors"
    cat "${cmd_computed_files}" | cut -d, -f2 | sort | uniq >"${uuids_list}"
fi
num_uuids=$(cat ${uuids_list} | wc -l)
log "Total UUIDs: ${num_uuids}"


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
print(os.path.relpath('${hash2uuids_index}', '$(dirname "${base_hash2uuids}")'))
")"
ln -sf "${rel_path}" "${base_hash2uuids}"


##
## Compute classifications
##
#if [ ! -s "${classifications_data}" ]; then
#    log "Computing classifications"
#    log_cc="${work_dir}/log.4.cc.txt"
#
#    "${script_cc}" -v -c "${config_cc}" --uuids-list "${uuids_list}" \
#                   --csv-header "${classifications_header}" \
#                   --csv-data "${classifications_data}" \
#                   2>&1 | tee "${log_cc}"
#fi


# TODO: (?) Fit new ball tree


log "Marking successful completion"
touch "${work_dir}/${complete_file}"
