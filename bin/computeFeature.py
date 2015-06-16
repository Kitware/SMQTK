#!/usr/bin/env python
"""
Compute a feature vector for a given file with a chosen ContentDescriptor type.
"""

import logging
import numpy
import optparse

from smqtk.data_rep.data_element_impl.file_element import DataFileElement
from smqtk.utils import bin_utils
from smqtk.utils.configuration import (
    ContentDescriptorConfiguration,
    DescriptorFactoryConfiguration,
)


def main():
    usage = "%prog [OPTIONS] INPUT_FILE"
    description = """\
Compute a descriptor vector for a given data file, outputting the generated
feature vector to standard out, or to an output file if one was specified (in
numpy format).
"""
    parser = bin_utils.SMQTKOptParser(usage, description=description)

    group_labels = optparse.OptionGroup(parser, "Configuration Labels")
    group_labels.add_option('-c', '--content-descriptor',
                            help='The descriptor type to use. This must be a '
                                 'type available in the system configuration')
    group_labels.add_option('-f', '--factory-type',
                            help='The DescriptorElementFactory configuration '
                                 'to use when computing the descriptor. This '
                                 'must be a type available in the system '
                                 'configuration.')
    parser.add_option_group(group_labels)

    group_optional = optparse.OptionGroup(parser, "Optional Parameters")
    group_optional.add_option('-l', '--list',
                              action='store_true', default=False,
                              help='List available descriptor types.')
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
    descriptor_label = opts.content_descriptor
    factory_label = opts.factory_type
    overwrite = opts.overwrite
    verbose = opts.verbose

    llevel = logging.DEBUG if verbose else logging.INFO
    bin_utils.initialize_logging(logging.getLogger(), llevel)
    log = logging.getLogger("main")

    if opts.list:
        log.info("")
        log.info("Available ContentDescriptor types:")
        log.info("")
        for dl in ContentDescriptorConfiguration.available_labels():
            log.info("\t%s", dl)
        log.info("")
        log.info("Available DescriptorElementFactory types:")
        log.info("")
        for df in DescriptorFactoryConfiguration.available_labels():
            log.info("\t%s", df)
        log.info("")
        exit(0)

    if len(args) == 0:
        log.error("Failed to provide an input file path")
        exit(1)
    if len(args) > 1:
        log.warning("More than one filepath provided as an argument. Only "
                    "computing for the first one.")

    input_filepath = args[0]
    data_element = DataFileElement(input_filepath)

    cd = ContentDescriptorConfiguration.new_inst(descriptor_label)
    factory = DescriptorFactoryConfiguration.new_inst(factory_label)
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
