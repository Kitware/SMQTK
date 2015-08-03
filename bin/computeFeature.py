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
    DataSetConfiguration,
    DetectorsAndDescriptorsConfiguration,
    QuantizationConfiguration,
    SimilarityNearestNeighborsConfiguration
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
    group_labels.add_option('-d', '--data-set',
        help="Data set to use for model generation.")
    group_labels.add_option('-c', '--content-descriptor',
        help='The descriptor type to use. This must be a '
        'type available in the system configuration')
    group_labels.add_option('-q', '--quantization',
        help="Quantizer for codebook generation.")
    group_labels.add_option('-s', '--snn',
        help="Similarity nearest neighbor configuration (FLANN, i.e.)")

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
    q_label = opts.quantization
    snn_label = opts.snn
    dset = opts.data_set

    overwrite = opts.overwrite
    verbose = opts.verbose

    llevel = logging.DEBUG if verbose else logging.INFO
    bin_utils.initialize_logging(logging.getLogger(), llevel)
    log = logging.getLogger("main")

    if opts.list:
        log.info("")
        log.info("Available Data Sets:")
        log.info("")
        for l in DataSetConfiguration.available_labels():
            log.info("\t%s" % l)
        log.info("")
        log.info("Available DetectorsAndDescriptorsConfigurations types:")
        log.info("")
        for dl in DetectorsAndDescriptorsConfiguration.available_labels():
            log.info("\t%s", dl)
        log.info("")
        log.info("Available Quantization types:")
        log.info("")
        for dl in QuantizationConfiguration.available_labels():
            log.info("\t%s", dl)
        log.info("")
        log.info("Available SimilarityNearestNeighborsConfigurations:")
        log.info("")
        for l in SimilarityNearestNeighborsConfiguration.available_labels():
            log.info("\t%s" % l)
        log.info("")

        exit(0)

    if len(args) == 0 and dset is None:
        log.error("Failed to provide an input file path or dataset")
        exit(1)
    if len(args) > 1:
        log.warning("More than one filepath provided as an argument. Only "
                    "computing for the first one.")

    if not q_label:
        log.error("Failed to provide quantization. Exiting.")
        exit(1)

    if not snn_label:
        log.error("Failed to provide indexer label. Exiting.")
        exit(1)

    input_filepath = args[0]
    data_element = DataFileElement(input_filepath)

    # Configure the feature detector and descriptor based on the json configuration file.
    log.info("Loading feature detection and descriptor %s." % descriptor_label)
    d_and_d = DetectorsAndDescriptorsConfiguration.new_inst(descriptor_label)
    # Detect and Describe
    log.info("Performing feature detection and description.")
    feat, desc = d_and_d.detect_and_describe(data_element)

    # Quantization -- ensure there is an existing quantization generated.
    quant = QuantizationConfiguration.new_inst(q_label)
    quant._descriptor_element_factory = d_and_d._descriptor_element_factory
    quant._data_element_type = d_and_d._data_element_type

    log.info("Checking to ensure there is an existing quantization for %s." % q_label)
    if quant.has_quantization:
        log.info("Success, quantization for %s exists." % q_label)
    else:
        log.error("Failed to find a quantization for %s. Exiting." % q_label)
        exit(1)

    # Load index
    log.info("Loading index for %s." % snn_label)
    snn = SimilarityNearestNeighborsConfiguration.new_inst(snn_label)
    vec = snn.build_histogram(feat, desc.vector(), quant._quantization_numpy)

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
