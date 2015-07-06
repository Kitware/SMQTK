#!/usr/bin/env python
"""
Generate model files for an ingest.
"""

import json
import logging
import multiprocessing.pool

from smqtk.utils import bin_utils
from smqtk.utils.configuration import (
    ConfigurationInterface,
    DataSetConfiguration,
    DescriptorFactoryConfiguration,
    ContentDescriptorConfiguration,
    IndexerConfiguration,
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



def list_available_fds_idxrs(log, ingest_config):
    """
    :type ingest_config: IngestConfiguration
    """
    log.info("")
    log.info("For ingest configuration '%s'...", ingest_config.label)
    log.info("")
    log.info("Available ContentDescriptor types:")
    log.info("")
    for l in ingest_config.get_available_descriptor_labels():
        log.info("\t%s" % l)
    log.info("")
    log.info("Available Indexer types:")
    log.info("")
    for l in ingest_config.get_available_indexer_labels():
        log.info("\t%s", l)
    log.info("")


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
    descr_fac_label = opts.descriptor_factory
    cd_label = opts.content_descriptor
    idxr_label = opts.indexer
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
        log.info("Available Descriptor Factory types:")
        for l in DescriptorFactoryConfiguration.available_labels():
            log.info("\t%s" % l)
        log.info("")
        log.info("Available ContentDescriptor types:")
        for l in ContentDescriptorConfiguration.available_labels():
            log.info("\t%s" % l)
        log.info("")
        log.info("Available Indexer types:")
        for l in IndexerConfiguration.available_labels():
            log.info("\t%s", l)
        log.info("")
        exit(0)

    # Check given labels
    fail = False
    if dset_label and dset_label not in DataSetConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated to an existing "
                  "data set configuration!", dset_label)
        fail = True
    if cd_label and cd_label not in ContentDescriptorConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated to an existing "
                  "content descriptor configuration!", cd_label)
        fail = True
    if idxr_label and idxr_label not in IndexerConfiguration.available_labels():
        log.error("Given label '%s' is NOT associated to an existing "
                  "indexer configuration!", idxr_label)
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

    log.info("Loading descriptor instance...")
    descriptor = ContentDescriptorConfiguration.new_inst(cd_label)
    # Generate any model files needed by the chosen descriptor
    descriptor.PARALLEL = parallel
    descriptor.generate_model(dset)

    # Don't do indexer model generation if a type was not provided
    if idxr_label:
        log.info("Loading indexer instance...")

        d_factory = DescriptorFactoryConfiguration.new_inst(descr_fac_label)
        indexer = IndexerConfiguration.new_inst(idxr_label)

        # It is not guaranteed that the feature computation method is doing
        # anything in parallel, but if it is, request that it perform serially
        # in order to allow multiple high-level feature computation jobs, else
        # we could be overrun with threads.
        descriptor.PARALLEL = 1
        # Using NonDaemonicPool because content_description that might to
        # parallel processing might use multiprocessing.Pool instances, too.
        # Pools don't usually allow daemonic processes, so this custom top-level
        # pool allows worker processes to spawn pools themselves.
        fmap = descriptor.compute_descriptor_async(
            dset, d_factory,
            parallel=parallel,
            pool_type=NonDaemonicPool
        )

        indexer.generate_model(fmap, parallel=parallel)


if __name__ == "__main__":
    main()
