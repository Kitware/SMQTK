"""
Retrieve the nearest neighbors of a set of UUIDs.

Input UIDs, specified in files input to the -u/--uuid-list option, are located
within the configured "descriptor_set" and then queried against the configured
"nn_index".  Thus, all UIDs provided in the input file should exist in the
configured "descriptor_set" otherwise a KeyError is raised.  Input descriptors
do not have to be necessarily indexed in the configured "nn_index".
"""
import logging
import os
import sys
from six.moves import zip

from smqtk.utils.cli import (
    basic_cli_parser,
    utility_main_helper,
)

from smqtk.algorithms import NearestNeighborsIndex
from smqtk.representation import DescriptorSet
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


__author__ = 'dan.lamanna@kitware.com'


def get_cli_parser():
    parser = basic_cli_parser(__doc__)
    parser.add_argument('-u', '--uuid-list',
                        default=None, metavar='PATH',
                        help='Path to list of UUIDs to calculate nearest '
                             'neighbors for. If empty, all UUIDs present '
                             'in the descriptor index will be used.')

    parser.add_argument('-n', '--num',
                        default=10, metavar='INT', type=int,
                        help='Number of maximum nearest neighbors to return '
                             'for each UUID, defaults to retrieving 10 nearest '
                             'neighbors. Set to 0 to retrieve all nearest '
                             'neighbors.')
    return parser


def get_default_config():
    return {
        'plugins': {
            'descriptor_set': make_default_config(DescriptorSet.get_impls()),
            'nn_index': make_default_config(NearestNeighborsIndex.get_impls())
        }
    }


def main():
    # Print help and exit if no arguments were passed
    if len(sys.argv) == 1:
        get_cli_parser().print_help()
        sys.exit(1)

    args = get_cli_parser().parse_args()
    config = utility_main_helper(get_default_config, args)

    log = logging.getLogger(__name__)
    log.debug('Showing debug messages.')

    #: :type: smqtk.representation.DescriptorSet
    descriptor_set = from_config_dict(
        config['plugins']['descriptor_set'], DescriptorSet.get_impls()
    )
    #: :type: smqtk.algorithms.NearestNeighborsIndex
    nearest_neighbor_index = from_config_dict(
        config['plugins']['nn_index'], NearestNeighborsIndex.get_impls()
    )

    # noinspection PyShadowingNames
    def nearest_neighbors(descriptor, n):
        if n == 0:
            n = len(nearest_neighbor_index)

        uuids, descriptors = nearest_neighbor_index.nn(descriptor, n)
        # Strip first result (itself) and create list of (uuid, distance)
        return list(zip([x.uuid() for x in uuids[1:]], descriptors[1:]))

    if args.uuid_list is not None and not os.path.exists(args.uuid_list):
        log.error('Invalid file list path: %s', args.uuid_list)
        exit(103)
    elif args.num < 0:
        log.error('Number of nearest neighbors must be >= 0')
        exit(105)

    if args.uuid_list is not None:
        with open(args.uuid_list, 'r') as infile:
            for line in infile:
                descriptor = descriptor_set.get_descriptor(line.strip())
                print(descriptor.uuid())
                for neighbor in nearest_neighbors(descriptor, args.num):
                    print('%s,%f' % neighbor)
    else:
        for (uuid, descriptor) in descriptor_set.iteritems():
            print(uuid)
            for neighbor in nearest_neighbors(descriptor, args.num):
                print('%s,%f' % neighbor)


if __name__ == '__main__':
    main()
