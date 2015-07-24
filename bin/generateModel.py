#!/usr/bin/env python
"""
Generate model files for an ingest.
"""
import json
import logging
import multiprocessing.pool
import numpy

from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorMemoryElement
from smqtk.utils import bin_utils
from smqtk.utils.configuration import (
    ConfigurationInterface,
    DataSetConfiguration,
    DetectorsAndDescriptorsConfiguration,
    QuantizationConfiguration,
    SimilarityNearestNeighborsConfiguration
)
from smqtk.utils.jsmin import jsmin

class NonDaemonicProcess (multiprocessing.Process):
    """ Same as normal processes, but force daemon flag to False """

    # noinspection PyMethodOverriding
    @multiprocessing.Process.daemon.setter
    def daemon(self, daemonic):
        """
        Set whether process is a daemon
        """
        assert self._popen is None, 'process has already started'
        # self._daemonic = daemonic


# noinspection PyAbstractClass
class NonDaemonicPool (multiprocessing.pool.Pool):
    """ Multiprocessing pool that uses Non-daemonic processes.

    This allows nested subprocesses.

    """

    Process = NonDaemonicProcess


def main():
    import optparse
    description = \
        "Generate the model for the given indexer type, using features " \
        "from the given feature descriptor type. We use configured valued in " \
        "the smqtk_config module and from the system configuration JSON file " \
        "(etc/system_config.json) unless otherwise specified by options to " \
        "this script. Specific ingest used is determined by the ingest type " \
        "provided (-t/--type)."
    parser = bin_utils.SMQTKOptParser(description=description)
    group_required = optparse.OptionGroup(parser, "Required Options")
    group_optional = optparse.OptionGroup(parser, "Optional")

    group_required.add_option('-d', '--data-set',
<<<<<<< HEAD
                              help="Data set to use for model generation.")
    group_required.add_option('-f', '--descriptor-factory',
                              help="Descriptor factory configuration label to "
                                   "use for descriptor storage.")
    group_required.add_option('-c', '--content-descriptor',
                              help="Content descriptor type for model and "
                                   "descriptor generation.")
    group_required.add_option('-i', '--indexer',
                              help="(Optional) Indexer type for model "
                                   "generation.")
=======
        help="Data set to use for model generation.")
    group_required.add_option('-c', '--content-detect-and-describe',
        help="Feature descriptor type for model and feature generation.")
    group_required.add_option('-q', '--quantization',
        help="Quantizer for codebook generation.")
    group_required.add_option('-s', '--snn',
        help="Similarity nearest neighbor configuration (FLANN, i.e.)")
>>>>>>> Refactored content descriptor into four ind. modules

    group_optional.add_option('--sys-json',
                              help="Custom system configuration JSON file to "
                                   "use. Otherwise we use the one specified in "
                                   "the smqtk_config module.")
    group_optional.add_option('-l', '--list',
                              action='store_true', default=False,
                              help="List available ingest configurations. If "
                                   "a valid ingest configuration has been "
                                   "specified, we list available "
                                   "FeatureDetector and Indexer configurations "
                                   "available.")
    group_optional.add_option('-t', '--threads', type=int, default=None,
                              help='Number of threads/processes to use for '
                                   'processing. By default we use all '
                                   'available cores/threads.')
    group_optional.add_option('-v', '--verbose', action='store_true',
                              default=False,
                              help='Add debug messaged to output logging.')

    parser.add_option_group(group_required)
    parser.add_option_group(group_optional)
    opts, args = parser.parse_args()

    bin_utils.initialize_logging(logging.getLogger(),
                                 logging.INFO - (10 * opts.verbose))
    log = logging.getLogger("main")

    dset_label = opts.data_set
    cd_label = opts.content_detect_and_describe
    q_label = opts.quantization
    snn_label = opts.snn
    parallel = opts.threads

    # Prep custom JSON configuration if one was given
    if opts.sys_json:
        with open(opts.sys_json) as json_file:
            json_config = json.loads(jsmin(json_file.read()))
        ConfigurationInterface.BASE_CONFIG = json_config['Ingests']

    if opts.list:
        log.info("")
        log.info("Available Data Sets:")
        for l in DataSetConfiguration.available_labels():
            log.info("\t%s" % l)
        log.info("")
        log.info("Available DetectorsAndDescriptorsConfigurations:")
        log.info("")
        for l in DetectorsAndDescriptorsConfiguration.available_labels():
            log.info("\t%s" % l)
        log.info("")
        log.info("Available QuantizationConfigurations:")
        log.info("")
        for l in QuantizationConfiguration.available_labels():
            log.info("\t%s" % l)
            log.info("")
        log.info("Available SimilarityNearestNeighborsConfigurations:")
        log.info("")
        for l in SimilarityNearestNeighborsConfiguration.available_labels():
            log.info("\t%s" % l)
        exit(0)

    # Check given labels
    fail = False
    if not dset_label:
        log.error("You have to provide me the name of a data set to use!")
        fail = True
    elif dset_label and dset_label not in DataSetConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated to an existing "
                  "data set configuration!", dset_label)
        fail = True
    if not cd_label:
        log.error("You have to provide me with a content descriptor label!")
        fail = True
    elif cd_label and cd_label not in DetectorsAndDescriptorsConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated to an existing "
                  "content descriptor configuration!", cd_label)
        fail = True
    if not q_label:
        log.error("You have to provide me with a quanization configuration!")
        fail = True
    elif q_label and q_label not in QuantizationConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated to an existing "
                  "quantization configuration!", q_label)
        fail = True
    if idxr_label and descr_fac_label and \
            descr_fac_label not in DescriptorFactoryConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated with an existing "
                  "descriptor factory configuration!", descr_fac_label)
        fail = True
    if fail:
        exit(1)
    del fail

    log.info("Loading data-set instance...")
    dset = DataSetConfiguration.new_inst(dset_label)

    # Detect and Describe
    log.info("Loading detect-and-describe instance...")
    dandd = DetectorsAndDescriptorsConfiguration.new_inst(cd_label)
    log.info("Running feature detection and description...")
    feat, desc = dandd.detect_and_describe(dset)

    # Quantize
    quant = QuantizationConfiguration.new_inst(q_label)
    quant.generate_quantization(desc)

    # Index
    log.info("Loading similiarity nearest neighbor instance...")
    snn = SimilarityNearestNeighborsConfiguration.new_inst(snn_label)
    snn.index(quant._quantization)

if __name__ == "__main__":
    main()
