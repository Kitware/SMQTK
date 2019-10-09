"""
Compute LSH hash codes based on the provided functor on all or specific
descriptors from the configured index given a file-list of UUIDs.

When using an input file-list of UUIDs, we require that the UUIDs of
indexed descriptors be strings, or equality comparable to the UUIDs' string
representation.

We update a key-value store with the results of descriptor hash computation. We
assume the keys of the store are the integer hash values and the values of the
store are ``frozenset`` instances of descriptor UUIDs (hashable-type objects).
We also assume that no other source is concurrently modifying this key-value
store due to the need to modify the values of keys.
"""
import logging
import os

from smqtk.algorithms import (
    LshFunctor,
)
from smqtk.compute_functions import compute_hash_codes
from smqtk.representation import (
    DescriptorSet,
    KeyValueStore,
)
from smqtk.utils import (
    cli,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


def uuids_for_processing(uuids, hash2uuids):
    """
    Determine descriptor UUIDs that need processing based on what's already in
    the given ``hash2uuids`` mapping, returning UUIDs that need processing.

    :param uuids: Iterable of descriptor UUIDs.
    :type uuids:

    :param hash2uuids: Existing mapping of computed hash codes to the UUIDs
        of descriptors that generated the hash.
    :type hash2uuids: smqtk.representation.KeyValueStore

    :return: Iterator over UUIDs to process
    :rtype: __generator[collections.Hashable]

    """
    log = logging.getLogger(__name__)
    already_there = frozenset(v for vs in hash2uuids.values() for v in vs)
    skipped = 0
    for uuid in uuids:
        if uuid not in already_there:
            yield uuid
        else:
            skipped += 1
    log.debug("Skipped %d UUIDs already represented in previous hash table",
              skipped)


def default_config():
    return {
        "utility": {
            "report_interval": 1.0,
            "use_multiprocessing": False,
        },
        "plugins": {
            "descriptor_set":
                make_default_config(DescriptorSet.get_impls()),
            "lsh_functor": make_default_config(LshFunctor.get_impls()),
            "hash2uuid_kvstore":
                make_default_config(KeyValueStore.get_impls()),
        },
    }


def cli_parser():
    parser = cli.basic_cli_parser(__doc__)

    g_io = parser.add_argument_group("I/O")
    g_io.add_argument("--uuids-list",
                      default=None, metavar="PATH",
                      help='Optional path to a file listing UUIDs of '
                           'descriptors to computed hash codes for. If '
                           'not provided we compute hash codes for all '
                           'descriptors in the configured descriptor index.')

    return parser


def main():
    args = cli_parser().parse_args()
    config = cli.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    #
    # Load configuration contents
    #
    uuid_list_filepath = args.uuids_list
    report_interval = config['utility']['report_interval']
    use_multiprocessing = config['utility']['use_multiprocessing']

    #
    # Checking input parameters
    #
    if (uuid_list_filepath is not None) and \
            not os.path.isfile(uuid_list_filepath):
        raise ValueError("UUIDs list file does not exist!")

    #
    # Loading stuff
    #
    log.info("Loading descriptor index")
    #: :type: smqtk.representation.DescriptorSet
    descriptor_set = from_config_dict(
        config['plugins']['descriptor_set'],
        DescriptorSet.get_impls()
    )
    log.info("Loading LSH functor")
    #: :type: smqtk.algorithms.LshFunctor
    lsh_functor = from_config_dict(
        config['plugins']['lsh_functor'],
        LshFunctor.get_impls()
    )
    log.info("Loading Key/Value store")
    #: :type: smqtk.representation.KeyValueStore
    hash2uuids_kvstore = from_config_dict(
        config['plugins']['hash2uuid_kvstore'],
        KeyValueStore.get_impls()
    )

    # Iterate either over what's in the file given, or everything in the
    # configured index.
    def iter_uuids():
        if uuid_list_filepath:
            log.info("Using UUIDs list file")
            with open(uuid_list_filepath) as f:
                for l in f:
                    yield l.strip()
        else:
            log.info("Using all UUIDs resent in descriptor index")
            for k in descriptor_set.keys():
                yield k

    #
    # Compute codes
    #
    log.info("Starting hash code computation")
    kv_update = {}
    for uuid, hash_int in \
            compute_hash_codes(uuids_for_processing(iter_uuids(),
                                                    hash2uuids_kvstore),
                               descriptor_set, lsh_functor,
                               report_interval,
                               use_multiprocessing, True):
        # Get original value in KV-store if not in update dict.
        if hash_int not in kv_update:
            kv_update[hash_int] = hash2uuids_kvstore.get(hash_int, set())
        kv_update[hash_int] |= {uuid}

    if kv_update:
        log.info("Updating KV store... (%d keys)" % len(kv_update))
        hash2uuids_kvstore.add_many(kv_update)

    log.info("Done")


if __name__ == '__main__':
    main()
