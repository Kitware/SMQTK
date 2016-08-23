"""
Compute a descriptor vector for a given data file, outputting the generated
feature vector to standard out, or to an output file if one was specified (in
numpy format).
"""

import logging
import os

import numpy

from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation import DescriptorElementFactory
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.utils import bin_utils, plugin


def default_config():
    return {
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "content_descriptor": plugin.make_config(get_descriptor_generator_impls()),
    }


def cli_parser():
    parser = bin_utils.basic_cli_parser(__doc__)

    parser.add_argument('--overwrite',
                        action='store_true', default=False,
                        help="Force descriptor computation even if an "
                             "existing descriptor vector was "
                             "discovered based on the given content "
                             "descriptor type and data combination.")
    parser.add_argument('-o', '--output-filepath',
                        help='Optional path to a file to output '
                             'feature vector to. Otherwise the feature '
                             'vector is printed to standard out. '
                             'Output is saved in numpy binary format '
                             '(.npy suffix recommended).')

    parser.add_argument("input_file",
                        nargs="?",
                        help="Data file to compute descriptor on")

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    output_filepath = args.output_filepath
    overwrite = args.overwrite

    if not args.input_file:
        log.error("Failed to provide an input file path")
        exit(1)
    elif not os.path.isfile(args.input_file):
        log.error("Given path does not point to a file.")
        exit(1)

    input_filepath = args.input_file
    data_element = DataFileElement(input_filepath)

    factory = DescriptorElementFactory.from_config(config['descriptor_factory'])
    #: :type: smqtk.algorithms.descriptor_generator.DescriptorGenerator
    cd = plugin.from_plugin_config(config['content_descriptor'],
                                   get_descriptor_generator_impls())
    descr_elem = cd.compute_descriptor(data_element, factory, overwrite)
    vec = descr_elem.vector()

    if vec is None:
        log.error("Failed to generate a descriptor vector for the input data!")

    if output_filepath:
        numpy.save(output_filepath, vec)
    else:
        # Construct string, because numpy
        s = []
        # noinspection PyTypeChecker
        for f in vec:
            s.append('%15f' % f)
        print ' '.join(s)


if __name__ == "__main__":
    main()
