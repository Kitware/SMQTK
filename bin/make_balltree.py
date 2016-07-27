#!/usr/bin/env python
#
# Script for building and saving the model for the ``SkLearnBallTreeHashIndex``
# ``HashIndex`` implementation.
#

import argparse, logging, cPickle, os, sys
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import SkLearnBallTreeHashIndex
from smqtk.utils.bin_utils import initialize_logging, report_progress
from smqtk.utils.bit_utils import int_to_bit_vector_large

initialize_logging(logging.getLogger(), logging.DEBUG)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("hash2uuids_fp", type=str)
parser.add_argument("bit_len", type=int)
parser.add_argument("leaf_size", type=int)
parser.add_argument("rand_seed", type=int)
parser.add_argument("balltree_model_fp", type=str)
args = parser.parse_args()

hash2uuids_fp = os.path.abspath(args.hash2uuids_fp)
bit_len = args.bit_len
leaf_size = args.leaf_size
rand_seed = args.rand_seed
balltree_model_fp = os.path.abspath(args.balltree_model_fp)

assert os.path.isfile(hash2uuids_fp), "Bad path: '%s'" % hash2uuids_fp
assert os.path.isdir(os.path.dirname(balltree_model_fp)), "Bad path: %s" % balltree_model_fp

log.debug("hash2uuids_fp    : %s", hash2uuids_fp)
log.debug("bit_len          : %d", bit_len)
log.debug("leaf_size        : %d", leaf_size)
log.debug("rand_seed        : %d", rand_seed)
log.debug("balltree_model_fp: %s", balltree_model_fp)


log.info("Loading hash2uuids table")
with open(hash2uuids_fp) as f:
    hash2uuids = cPickle.load(f)

log.info("Computing hash-code vectors")
hash_vectors = []  #[int_to_bit_vector_large(h, bit_len) for h in hash2uuids]
rs = [0] * 7
for h in hash2uuids:
    hash_vectors.append( int_to_bit_vector_large(h, bit_len) )
    report_progress(log.debug, rs, 1.)

log.info("Initializing ball tree")
btree = SkLearnBallTreeHashIndex(balltree_model_fp, leaf_size, rand_seed)

log.info("Building ball tree")
btree.build_index(hash_vectors)
