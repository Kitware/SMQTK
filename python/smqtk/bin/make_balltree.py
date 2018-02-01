"""
Script for building and saving the model for the ``SkLearnBallTreeHashIndex``
implementation of ``HashIndex``.
"""

import logging, os
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
import smqtk.representation
import smqtk.utils.bin_utils
import smqtk.utils.bit_utils
import smqtk.utils.plugin

try:
    import cPickle as pickle
except ImportError:
    import pickle


def default_config():
    return {
        "hash2uuid_kv_store": smqtk.utils.plugin.make_config(
            smqtk.representation.get_key_value_store_impls()
        ),
        "sklearn_balltree": SkLearnBallTreeHashIndex.get_default_config(),
        "itq_bit_length": 256,
    }


def cli_parser():
    parser = smqtk.utils.bin_utils.basic_cli_parser(__doc__)
    return parser


def main():
    args = cli_parser().parse_args()
    config = smqtk.utils.bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    # Loading from configurations
    bit_len = int(config['itq_bit_length'])
    log.info("Loading hash2uuid KeyValue store")
    #: :type: smqtk.representation.KeyValueStore
    hash2uuid_kv_store = smqtk.utils.plugin.from_plugin_config(
        config['hash2uuid_kv_store'],
        smqtk.representation.get_key_value_store_impls()
    )
    log.info("Initializing ball tree")
    btree = SkLearnBallTreeHashIndex.from_config(config['sklearn_balltree'])

    log.info("Computing hash-code vectors")
    hash_vectors = []
    reporter = smqtk.utils.bin_utils.ProgressReporter(log.debug, 1.0)
    reporter.start()
    for h in hash2uuid_kv_store.keys():
        hash_vectors.append(
            smqtk.utils.bit_utils.int_to_bit_vector_large(h, bit_len))
        reporter.increment_report()
    reporter.report()

    log.info("Building ball tree index")
    btree.build_index(hash_vectors)


if __name__ == '__main__':
    main()
