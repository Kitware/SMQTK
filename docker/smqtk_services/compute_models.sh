#!/usr/bin/env bash
#
# Compute necessary descriptors and model files for Nearest-Neighbor and IQR
# service applications.
#
set -e
. /smqtk/install/setup_smqtk.sh

IMAGE_LIST=/app/models/imagelist.txt

CMD_CONFIG=/app/configs/train.cmd.config.json
CMD_LOG=/logs/train.cmd.log
CMD_PROCESSED=/app/models/cmd.processed.csv

ITQ_CONFIG=/app/configs/train.itq.config.json
ITQ_LOG=/logs/train.itq.log

CHC_CONFIG=/app/configs/train.chc.config.json
CHC_LOG=/logs/train.chc.log
HASH2UUIDS=/app/models/lsh/hash2uuids.32bit.pickle

BALLTREE_LOG=/logs/train.hash_index.balltree.32bit.log
BALLTREE_BITSIZE=32
BALLTREE_MODEL=/app/models/lsh/hash_index.balltree.npz


# Compute descriptors -> PostgreSQL
echo "#########################################"
echo "# Creating image list of files in /data # "
echo "#########################################"
find /data -type f >"${IMAGE_LIST}"
if [ "$(cat "${IMAGE_LIST}" | wc -l)" -eq 0 ]
then
    echo "ERROR: Failed to find any files in /data."
    echo "       Did you forget to mount the volume?"
    exit 1
fi

echo "#################################"
echo "# Computing content descriptors #"
echo "#################################"
compute_many_descriptors.py \
    -vc "${CMD_CONFIG}" \
    -b 0 \
    --check-image \
    -f "${IMAGE_LIST}" \
    -p "${CMD_PROCESSED}" \
    2>&1 | tee "${CMD_LOG}"


# Compute ITQ models -> /app/itq/{mean_vec,rotation}.npy
echo "######################"
echo "# Training ITQ model #"
echo "######################"
train_itq.py \
    -vc "${ITQ_CONFIG}" \
    2>&1 | tee "${ITQ_LOG}"


# Hash2uuids model -> /app/lsh/hash2uuids.pickle
echo "###################################"
echo "# Computing descriptor hash codes #"
echo "###################################"
compute_hash_codes.py \
    -vc "${CHC_CONFIG}" \
    --output-hash2uuids "${HASH2UUIDS}" \
    2>&1 | tee "${CHC_LOG}"


# OPTIONAL( BallTree ) -> /app/hash_index/balltree.npz
echo "####################################"
echo "# Building BallTree hashcode index #"
echo "####################################"
python -c "
import logging, cPickle
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import SkLearnBallTreeHashIndex
from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.bit_utils import int_to_bit_vector

initialize_logging(logging.getLogger(), logging.DEBUG)

with open('${HASH2UUIDS}') as f:
    hash2uuids = cPickle.load(f)
hash_vectors = [int_to_bit_vector(h, ${BALLTREE_BITSIZE}) for h in hash2uuids]

btree = SkLearnBallTreeHashIndex('${BALLTREE_MODEL}', random_seed=0)
btree.build_index(hash_vectors)
" 2>&1 | tee "${BALLTREE_LOG}"
