#!/usr/bin/env python
"""
Compute a feature vector for a given file with a chosen FeatureDescriptor type.
"""

import logging
import numpy

from SMQTK.utils import bin_utils
from SMQTK.utils.configuration import IngestConfiguration


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
    parser.add_option('-i', '--ingest-type',
                      help='The ingest configuration to use.')
    parser.add_option('-d', '--descriptor-type',
                      help='The descriptor type to use. This must be a type '
                           'available in the given configuration')
    parser.add_option('-o', '--output-filepath',
                      help='Optional path to a file to output feature vector '
                           'to. Otherwise the feature vector is printed to '
                           'standard out. Output is saved in numpy binary '
                           'format (.npy suffix recommended).')
    parser.add_option('-l', '--list', action='store_true', default=False,
                      help='List available types. If an ingest type was not '
                           'provided, we list available ingest types. If an '
                           'ingest type has been given, we list available '
                           'feature descriptor types available within that '
                           'ingest.')
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                      help='Print additional debugging messages. All logging '
                           'goes to standard error.')
    opts, args = parser.parse_args()

    output_filepath = opts.output_filepath
    ingest_type = opts.ingest_type
    descriptor_type = opts.descriptor_type
    verbose = opts.verbose

    bin_utils.initializeLogging(logging.getLogger(),
                                logging.INFO - (10*verbose))
    log = logging.getLogger("main")

    if opts.list:
        if ingest_type is None:
            log.info("")
            log.info("Available Ingest configurations:")
            log.info("")
            for il in IngestConfiguration.available_ingest_labels():
                log.info("\t%s", il)
            log.info("")
            exit(0)

        else:
            ic = IngestConfiguration(ingest_type)
            log.info("")
            log.info("Available Descriptor types for ingest '%s':", ingest_type)
            log.info("")
            for dl in ic.get_available_descriptor_labels():
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

    ingest_config = IngestConfiguration(ingest_type)
    ic = ingest_config.new_ingest_instance()
    data_element = ic.DATA_FILE_TYPE(input_filepath)
    fd = ingest_config.new_descriptor_instance(descriptor_type)
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
