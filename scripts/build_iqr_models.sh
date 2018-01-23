#!/usr/bin/env bash
#
# Standard process of building IQR-required descriptors+models from scratch
# and existing configs.
#
# This assumes the use of the LSH nearest-neighbor index as it builds ITQ model.
# This could be triggered in the future by introspecting configs?
#
set -e

# PARAMETERS ###################################################################

IMAGE_DIR="images.tiles"

SMQTK_CMD_CONFIG="config.cmd.json"
SMQTK_CMD_BATCH_SIZE=1000
SMQTK_CMD_PROCESSED_CSV="cmd.processed.csv"

SMQTK_ITQ_TRAIN_CONFIG="config.train_itq.json"
SMQTK_ITQ_BIT_SIZE=256

SMQTK_HCODE_CONFIG="config.chc.json"
SMQTK_HCODE_PICKLE="models/smqtk/lsh.hash2uuids.pickle"

SMQTK_HCODE_BTREE_LEAFSIZE=1000
SMQTK_HCODE_BTREE_RAND=0
SMQTK_HCODE_BTREE_OUTPUT="models/smqtk/hash_index_btree.npz"

# DON'T MODIFY BELOW HERE ######################################################

# Create list of image files
IMAGE_DIR_FILELIST="${IMAGE_DIR}.filelist.txt"
find "${IMAGE_DIR}/" -type f >"${IMAGE_DIR_FILELIST}"

# Compute descriptors
compute_many_descriptors \
    -v -b ${SMQTK_CMD_BATCH_SIZE} --check-image -c "${SMQTK_CMD_CONFIG}" \
    -f "${IMAGE_DIR_FILELIST}" -p "${SMQTK_CMD_PROCESSED_CSV}"

# Train ITQ models
train_itq -vc "${SMQTK_ITQ_TRAIN_CONFIG}"

# Compute hash codes for descriptors
compute_hash_codes \
    -vc "${SMQTK_HCODE_CONFIG}" \
    --output-hash2uuids "${SMQTK_HCODE_PICKLE}"

# Compute balltree hash index
make_balltree "${SMQTK_HCODE_PICKLE}" ${SMQTK_ITQ_BIT_SIZE} \
    ${SMQTK_HCODE_BTREE_LEAFSIZE} ${SMQTK_HCODE_BTREE_RAND} \
    ${SMQTK_HCODE_BTREE_OUTPUT}
