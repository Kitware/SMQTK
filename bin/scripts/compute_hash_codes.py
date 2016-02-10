#!/usr/bin/env python
import cPickle
import logging
import os

from smqtk.algorithms import (
    get_lsh_functor_impls,
)
from smqtk.compute_functions import compute_hash_codes
from smqtk.representation import (
    get_descriptor_index_impls,
)
from smqtk.utils import (
    bin_utils,
    file_utils,
    plugin,
)


__author__ = "paul.tunison@kitware.com"


def uuids_for_processing(uuids, hash2uuids):
    """
    Determine descriptor UUIDs that need processing based on what's already in
    the given ``hash2uuids`` mapping, returning UUIDs that need processing.

    :param uuids: Iterable of descriptor UUIDs.
    :type uuids:

    :param hash2uuids: Existing mapping of computed hash codes to the UUIDs
        of descriptors that generated the hash.
    :type hash2uuids: dict[int|long, set[collections.Hashable]]

    :return:
    :rtype:

    """
    already_there = frozenset(v for vs in hash2uuids.itervalues() for v in vs)
    for uuid in uuids:
        if uuid not in already_there:
            yield uuid


def default_config():
    return {
        "utility": {
            "uuid_list_filepath": None,
            "hash2uuids_input_filepath": None,
            "hash2uuids_output_filepath": None,
            "report_interval": 1.0,
            "use_multiprocessing": False,
            "pickle_protocol": -1,
        },
        "plugins": {
            "descriptor_index": plugin.make_config(get_descriptor_index_impls()),
            "lsh_functor": plugin.make_config(get_lsh_functor_impls()),
        },
    }


def main():
    description = """
    Compute LSH hash codes based on the provided functor on specific
    descriptors from the configured index given a file-list of UUIDs.

    Due to using an input file-list of UUIDs, we require that the UUIDs of
    indexed descriptors be strings, or equality comparable to the UUIDs' string
    representation.

    This script can be used to live update the ``hash2uuid_cache_filepath``
    model file for the ``LSHNearestNeighborIndex`` algorithm as output
    dictionary format is the same as used by that implementation.
    """
    args, config = bin_utils.utility_main_helper(default_config(),
                                                 description)
    log = logging.getLogger(__name__)

    #
    # Load configuration contents
    #
    uuid_list_filepath = config['utility']['uuid_list_filepath']
    hash2uuids_input_filepath = config['utility']['hash2uuids_input_filepath']
    hash2uuids_output_filepath = config['utility']['hash2uuids_output_filepath']
    report_interval = config['utility']['report_interval']
    use_multiprocessing = config['utility']['use_multiprocessing']
    pickle_protocol = config['utility']['pickle_protocol']

    #
    # Checking parameters
    #
    if uuid_list_filepath is None or not os.path.isfile(uuid_list_filepath):
        raise ValueError("No UUID list file given!")

    #
    # Loading stuff
    #
    log.info("Loading descriptor index")
    #: :type: smqtk.representation.DescriptorIndex
    descriptor_index = plugin.from_plugin_config(
        config['plugins']['descriptor_index'],
        get_descriptor_index_impls()
    )
    log.info("Loading LSH functor")
    #: :type: smqtk.algorithms.LshFunctor
    lsh_functor = plugin.from_plugin_config(
        config['plugins']['lsh_functor'],
        get_lsh_functor_impls()
    )

    log.info("Loading UUIDs list")
    def iter_uuids():
        with open(uuid_list_filepath) as f:
            for l in f:
                yield l.strip()

    # load map if it exists, else start with empty dictionary
    if hash2uuids_input_filepath and os.path.isfile(hash2uuids_input_filepath):
        log.info("Loading hash2uuids mapping")
        with open(hash2uuids_input_filepath) as f:
            hash2uuids = cPickle.load(f)
    else:
        log.info("Creating new hash2uuids mapping for output")
        hash2uuids = {}

    #
    # Compute codes
    #
    compute_hash_codes(
        uuids_for_processing(iter_uuids(), hash2uuids),
        descriptor_index,
        lsh_functor,
        hash2uuids,
        report_interval=report_interval,
        use_mp=use_multiprocessing,
    )

    #
    # Output results
    #
    tmp_output_filepath = hash2uuids_output_filepath+'.WRITING'
    log.info("Writing hash-to-uuids map to disk: %s", tmp_output_filepath)
    file_utils.safe_create_dir(os.path.dirname(hash2uuids_output_filepath))
    with open(tmp_output_filepath, 'wb') as f:
        cPickle.dump(hash2uuids, f, pickle_protocol)
    log.info("Moving on top of input: %s", hash2uuids_output_filepath)
    os.rename(tmp_output_filepath, hash2uuids_output_filepath)
    log.info("Done")


if __name__ == '__main__':
    main()
