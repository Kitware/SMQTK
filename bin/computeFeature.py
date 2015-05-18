#!/usr/bin/env python
"""
Compute a feature vector for a given file with a chosen ContentDescriptor type.
"""

import logging
import numpy

from smqtk.data_rep.data_element_impl.file_element import FileElement
from smqtk.utils import bin_utils
from smqtk.utils.configuration import ContentDescriptorConfiguration


def main():
    usage = "%prog [OPTIONS] INPUT_FILE"
    description = "Compute a feature vector for a given data file, outputting " \
                  "the generated feature vector to standard out, or to an " \
                  "output file if one was specified.\n" \
                  "\n" \
                  "An ingest " \
                  "configuration must be specified for the purpose of " \
                  "identifying which model files to use (assuming a given " \
                  "descriptor has/uses model files). The ingest configuration " \
                  "also informs where to put temporary working files. "
    parser = bin_utils.SMQTKOptParser(usage, description=description)
    parser.add_option('-c', '--content-descriptor',
                      help='The descriptor type to use. This must be a type '
                           'available in system configuration')
    parser.add_option('-o', '--output-filepath',
                      help='Optional path to a file to output feature vector '
                           'to. Otherwise the feature vector is printed to '
                           'standard out. Output is saved in numpy binary '
                           'format (.npy suffix recommended).')
    parser.add_option('-l', '--list', action='store_true', default=False,
                      help='List available descriptor types.')
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                      help='Print additional debugging messages. All logging '
                           'goes to standard error.')
    opts, args = parser.parse_args()

    output_filepath = opts.output_filepath
    descriptor_type = opts.content_descriptor
    verbose = opts.verbose

    llevel = logging.DEBUG if verbose else logging.INFO
    bin_utils.initializeLogging(logging.getLogger(), llevel)
    log = logging.getLogger("main")

    if opts.list:
        log.info("")
        log.info("Available ContentDescriptor types:")
        log.info("")
        for dl in ContentDescriptorConfiguration.available_labels():
            log.info("\t%s", dl)
        log.info("")
        exit(0)

    if len(args) == 0:
        log.error("Failed to provide an input file path")
        exit(1)
    if len(args) > 1:
        log.warning("More than one filepath provided as an argument. Only "
                    "computing for the first one.")

    input_filepath = args[0]
    data_element = FileElement(input_filepath)

    fd = ContentDescriptorConfiguration.new_inst(descriptor_type)
    feat = fd.compute_feature(data_element)

    if output_filepath:
        numpy.save(output_filepath, feat)
    else:
        # Construct string, because numpy
        s = []
        for f in feat:
            s.append('%15f' % f)
        print ' '.join(s)


if __name__ == "__main__":
    main()
