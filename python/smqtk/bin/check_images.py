"""
Validate a list of images returning the filepaths and UUIDs of only the
valid images.
"""
import itertools
import functools
import logging
import os
import sys

from smqtk.utils.bin_utils import (
    basic_cli_parser,
    initialize_logging,
)

from smqtk.utils.image_utils import is_valid_element
from smqtk.utils import parallel


__author__ = 'dan.lamanna@kitware.com'


def get_cli_parser():
    parser = basic_cli_parser(__doc__, configuration_group=False)
    g_required = parser.add_argument_group("Required Arguments")
    g_required.add_argument('-f', '--file-list',
                            type=str, default=None, metavar='PATH',
                            help='Path to a file that lists data file paths. '
                                 'Paths in this file may be relative, but '
                                 'will at some point be coerced into absolute '
                                 'paths based on the current working '
                                 'directory.')
    return parser


def main():
    # Print help and exit if no arguments were passed
    if len(sys.argv) == 1:
        get_cli_parser().print_help()
        sys.exit(1)

    args = get_cli_parser().parse_args()
    llevel = logging.INFO if not args.verbose else logging.DEBUG
    initialize_logging(logging.getLogger('smqtk'), llevel)
    initialize_logging(logging.getLogger('__main__'), llevel)

    log = logging.getLogger(__name__)
    log.debug('Showing debug messages.')

    if args.file_list is not None and not os.path.exists(args.file_list):
        log.error('Invalid file list path: %s', args.file_list)
        exit(103)

    with open(args.file_list) as infile:
        valid_element_func = functools.partial(is_valid_element,
                                               check_image=True)
        valid_images = parallel.parallel_map(valid_element_func,
                                             itertools.imap(str.strip, infile),
                                             name='check-file-type',
                                             use_multiprocessing=True)

        for dfe in valid_images:
            if dfe is not None:
                print('%s,%s' % (dfe._filepath, dfe.uuid()))


if __name__ == '__main__':
    main()
