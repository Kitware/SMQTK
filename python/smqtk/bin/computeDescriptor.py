"""
Compute a descriptor vector for a given data file, outputting the generated
feature vector to standard out, or to an output file if one was specified (in
numpy format).
"""
from __future__ import print_function
import logging
import os

import numpy

from smqtk.algorithms import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation import DescriptorElementFactory
from smqtk.utils import cli
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


def default_config():
    return {
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "content_descriptor":
            make_default_config(DescriptorGenerator.get_impls()),
    }


def cli_parser():
    parser = cli.basic_cli_parser(__doc__)

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


def generate_vector(log, generator, data_element, factory, overwrite):
    descr_elem = generator.generate_one_element(
        data_element, descr_factory=factory, overwrite=overwrite
    )
    vec = descr_elem.vector()

    if vec is None:
        log.error("Failed to generate a descriptor vector for the input data!")

    return vec


def main():
    parser = cli_parser()
    args = parser.parse_args()
    config = cli.utility_main_helper(default_config, args)
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
    cd = from_config_dict(config['content_descriptor'],
                          DescriptorGenerator.get_impls())

    vec = generate_vector(log, cd, data_element, factory, overwrite)

    if output_filepath:
        numpy.save(output_filepath, vec)
    else:
        # Construct string, because numpy
        s = []
        # noinspection PyTypeChecker
        for f in vec:
            s.append('%15f' % f)
        print(' '.join(s))


if __name__ == "__main__":
    main()
