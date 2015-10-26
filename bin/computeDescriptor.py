#!/usr/bin/env python
"""
Compute a descriptor vector for a given file given a configuration that
specifies what descriptor generator to use, and where to store generated
DescriptorElements.
"""

import json
import logging
import argparse
import os

import numpy

from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation import DescriptorElementFactory
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.utils import bin_utils, plugin


def default_config():
    return {
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "content_descriptor": plugin.make_config(get_descriptor_generator_impls),
    }

def cli_parser():
    usage = "%prog [OPTIONS] INPUT_FILE"
    description = """\
Compute a descriptor vector for a given data file, outputting the generated
feature vector to standard out, or to an output file if one was specified (in
numpy format).
"""
    #parser = bin_utils.SMQTKOptParser(usage, description=description)
    parser = argparse.ArgumentParser(description=description)

    group_configuration = parser.add_argument_group("Configuration")
    group_configuration.add_argument('-c', '--config',
                            default=None,
                            help='Path to the JSON configuration file.')
    group_configuration.add_argument('--output-config',
                            default=None,
                            help='Optional path to output default JSON '
                                 'configuration to.')
    # parser.add_option_group(group_labels)

    group_optional = parser.add_argument_group("Optional Parameters")
    group_optional.add_argument('--overwrite',
                              action='store_true', default=False,
                              help="Force descriptor computation even if an "
                                   "existing descriptor vector was discovered "
                                   "based on the given content descriptor type "
                                   "and data combination.")
    group_optional.add_argument('-o', '--output-filepath',
                              help='Optional path to a file to output feature '
                                   'vector to. Otherwise the feature vector is '
                                   'printed to standard out. Output is saved '
                                   'in numpy binary format (.npy suffix '
                                   'recommended).')
    group_optional.add_argument('-v', '--verbose',
                              action='store_true', default=False,
                              help='Print additional debugging messages. All '
                                   'logging goes to standard error.')

    parser.add_argument("input_file",help="Data file to compute descriptor on")

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    output_filepath = args.output_filepath
    overwrite = args.overwrite
    verbose = args.verbose

    llevel = logging.DEBUG if verbose else logging.INFO
    bin_utils.initialize_logging(logging.getLogger(), llevel)
    log = logging.getLogger("main")

    bin_utils.output_config(args.output_config, default_config(), log)

    if not args.config:
        log.error("No configuration provided")
        exit(1)
    elif not os.path.isfile(args.config):
        log.error("Configuration file path not valid.")
        exit(1)

    if not args.input_file:
        log.error("Failed to provide an input file path")
        exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)

    input_filepath = args.input_file
    data_element = DataFileElement(input_filepath)

    factory = DescriptorElementFactory.from_config(config['descriptor_factory'])
    #: :type: smqtk.descriptor_generator.DescriptorGenerator
    cd = plugin.from_plugin_config(config['content_descriptor'],
                                   get_descriptor_generator_impls)
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
