#!/usr/bin/env python
"""
Compute a feature vector for a given file with a chosen DescriptorGenerator type.
"""

import json
import logging
import optparse
import os

import numpy

from smqtk.representation.data_element_impl.file_element import DataFileElement
from smqtk.representation import DescriptorElementFactory
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.utils import bin_utils, plugin


def default_config():
    return {
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "content_descriptor": plugin.make_config(get_descriptor_generator_impls),
    }


def main():
    usage = "%prog [OPTIONS] INPUT_FILE"
    description = """\
Compute a descriptor vector for a given data file, outputting the generated
feature vector to standard out, or to an output file if one was specified (in
numpy format).
"""
    parser = bin_utils.SMQTKOptParser(usage, description=description)

    group_labels = optparse.OptionGroup(parser, "Configuration")
    group_labels.add_option('-c', '--config',
                            default=None,
                            help='Path to the JSON configuration file.')
    group_labels.add_option('--output-config',
                            default=None,
                            help='Optional path to output default JSON '
                                 'configuration to.')
    parser.add_option_group(group_labels)

    group_optional = optparse.OptionGroup(parser, "Optional Parameters")
    group_optional.add_option('--overwrite',
                              action='store_true', default=False,
                              help="Force descriptor computation even if an "
                                   "existing descriptor vector was discovered "
                                   "based on the given content descriptor type "
                                   "and data combination.")
    group_optional.add_option('-o', '--output-filepath',
                              help='Optional path to a file to output feature '
                                   'vector to. Otherwise the feature vector is '
                                   'printed to standard out. Output is saved '
                                   'in numpy binary format (.npy suffix '
                                   'recommended).')
    group_optional.add_option('-v', '--verbose',
                              action='store_true', default=False,
                              help='Print additional debugging messages. All '
                                   'logging goes to standard error.')
    parser.add_option_group(group_optional)

    opts, args = parser.parse_args()

    output_filepath = opts.output_filepath
    overwrite = opts.overwrite
    verbose = opts.verbose

    llevel = logging.DEBUG if verbose else logging.INFO
    bin_utils.initialize_logging(logging.getLogger(), llevel)
    log = logging.getLogger("main")

    bin_utils.output_config(opts.output_config, default_config(), log)

    if not opts.config:
        log.error("No configuration provided")
        exit(1)
    elif not os.path.isfile(opts.config):
        log.error("Configuration file path not valid.")
        exit(1)

    if len(args) == 0:
        log.error("Failed to provide an input file path")
        exit(1)
    if len(args) > 1:
        log.warning("More than one filepath provided as an argument. Only "
                    "computing for the first one.")

    with open(opts.config, 'r') as f:
        config = json.load(f)

    input_filepath = args[0]
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
