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

LOG_DIR="logs"

IMAGE_DIR="images"

# This config can optionally exist. If it does not exist, no image
# transformation processing (e.g. tiling, etc.) is done.
SMQTK_IMAGE_TRANSFORM_CONFIG="configs/generate_image_transform.tiles.json"
SMQTK_IMAGE_TRANSFORM_OUTPUT_DIR="image_tiles"

SMQTK_CMD_CONFIG="configs/compute_many_descriptors.json"
SMQTK_CMD_BATCH_SIZE=1000
SMQTK_CMD_PROCESSED_CSV="models/cmd.processed.csv"

SMQTK_ITQ_TRAIN_CONFIG="configs/train_itq.json"

SMQTK_HCODE_CONFIG="configs/compute_hash_codes.json"
SMQTK_HCODE_PICKLE="models/lsh.hash2uuids.pickle"

SMQTK_HCODE_BTREE_CONFIG="configs/make_balltree.json"

# DON'T MODIFY BELOW HERE ######################################################
mkdir -p "${SMQTK_IMAGE_TRANSFORM_OUTPUT_DIR}" \
         "${LOG_DIR}"

# Create list of image files and followable links.
IMAGE_DIR_FILELIST="${IMAGE_DIR}.filelist.txt"
find "${IMAGE_DIR}/" -type f -follow >"${IMAGE_DIR_FILELIST}"

# Create generate image transforms as appropriate.
if [ -f "${SMQTK_IMAGE_TRANSFORM_CONFIG}" ]
then
  for FPATH in $(cat "${IMAGE_DIR_FILELIST}")
  do
    echo "Progessing: ${FPATH}"
    generate_image_transform -c "${SMQTK_IMAGE_TRANSFORM_CONFIG}" \
      -i "${FPATH}" \
      -o "${SMQTK_IMAGE_TRANSFORM_OUTPUT_DIR}"
  done
  # Rewrite image list for tiles generated.
  find "${SMQTK_IMAGE_TRANSFORM_OUTPUT_DIR}" -type f -follow >"${IMAGE_DIR_FILELIST}"
fi

# Compute descriptors
compute_many_descriptors \
    -v -b ${SMQTK_CMD_BATCH_SIZE} --check-image -c "${SMQTK_CMD_CONFIG}" \
    -f "${IMAGE_DIR_FILELIST}" -p "${SMQTK_CMD_PROCESSED_CSV}" \
    2>&1 | tee "${LOG_DIR}/compute_many_descriptors.log"

# Train ITQ models
train_itq -vc "${SMQTK_ITQ_TRAIN_CONFIG}" \
    2>&1 | tee "${LOG_DIR}/train_itq.log"

# Compute hash codes for descriptors
compute_hash_codes \
    -vc "${SMQTK_HCODE_CONFIG}" \
    2>&1 | tee "${LOG_DIR}/compute_hash_codes.log"

# Compute balltree hash index
make_balltree \
    -vc "${SMQTK_HCODE_BTREE_CONFIG}" \
    2>&1 | tee "${LOG_DIR}/make_balltree.log"
