"""
Validate a list of images returning the filepaths and UUIDs of only the
valid images, or optionally, only the invalid images.
"""
import itertools
import logging
import os
import sys

from smqtk.utils.bin_utils import (
    basic_cli_parser,
    initialize_logging,
)
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils.image_utils import is_valid_element
from smqtk.utils import parallel


__author__ = 'dan.lamanna@kitware.com'


def get_cli_parser():
    parser = basic_cli_parser(__doc__, configuration_group=False)

    parser.add_argument('-i', '--invert',
                        default=False, action='store_true',
                        help='Invert results, showing only invalid images.')

    g_required = parser.add_argument_group("Required Arguments")
    g_required.add_argument('-f', '--file-list',
                            type=str, default=None, metavar='PATH',
                            help='Path to a file that lists data file paths.')
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

    def check_image(image_path):
        if not os.path.exists(image_path):
            log.warn('Invalid image path given (does not exist): %s', image_path)
            return (False, False)
        else:
            dfe = DataFileElement(image_path)
            return (is_valid_element(dfe, check_image=True), dfe)

    with open(args.file_list) as infile:
        checked_images = parallel.parallel_map(check_image,
                                               itertools.imap(str.strip, infile),
                                               name='check-image-validity',
                                               use_multiprocessing=True)

        for (is_valid, dfe) in checked_images:
            if dfe != False: # in the case of a non-existent file
                if (is_valid and not args.invert) or (not is_valid and args.invert):
                    print('%s,%s' % (dfe._filepath, dfe.uuid()))


if __name__ == '__main__':
    main()
