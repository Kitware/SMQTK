#!/usr/bin/env python
"""
Generate model files for an ingest.
"""

import json
import logging
import multiprocessing.pool

from SMQTK.content_description import FeatureDescriptor
from SMQTK.utils import bin_utils
from SMQTK.utils.configuration import IngestConfiguration
from SMQTK.utils.jsmin import jsmin


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


def list_ingest_labels(log):
    log.info("")
    log.info("Available ingest labels:")
    log.info("")
    for l in IngestConfiguration.available_ingest_labels():
        log.info("\t%s", l)
    log.info("")


def list_available_fds_idxrs(log, ingest_config):
    """
    :type ingest_config: IngestConfiguration
    """
    log.info("")
    log.info("For ingest configuration '%s'...", ingest_config.label)
    log.info("")
    log.info("Available FeatureDescriptor types:")
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
    groupRequired = optparse.OptionGroup(parser, "Required Options")
    groupOptional = optparse.OptionGroup(parser, "Optional")

    groupRequired.add_option('-i', '--ingest',
                             help="Ingest configuration to use.")
    groupRequired.add_option('-d', '--feature-descriptor',
                             help="Feature descriptor type for model and "
                                  "feature generation.")
    groupRequired.add_option('-I', '--indexer',
                             help="Indexer type for model generation.")

    groupOptional.add_option('--sys-json',
                             help="Custom system configuration JSON file to "
                                  "use. Otherwise we use the one specified in "
                                  "the smqtk_config module.")
    groupOptional.add_option('-l', '--list', action='store_true', default=False,
                             help="List available ingest configurations. If "
                                  "a valid ingest configuration has been "
                                  "specified, we list available "
                                  "FeatureDetector and Indexer configurations "
                                  "available.")
    groupOptional.add_option('-t', '--threads', type=int, default=None,
                             help='Number of threads/processes to use for '
                                  'processing.')
    groupOptional.add_option('-v', '--verbose', action='store_true',
                             default=False,
                             help='Add debug messaged to output logging.')

    parser.add_option_group(groupRequired)
    parser.add_option_group(groupOptional)
    opts, args = parser.parse_args()

    bin_utils.initializeLogging(logging.getLogger(),
                                logging.INFO - (10*opts.verbose))
    log = logging.getLogger("main")

    ingest_label = opts.ingest
    fd_label = opts.feature_descriptor
    idxr_label = opts.indexer
    parallel = opts.threads

    # Prep custom JSON configuration if one was given
    if opts.sys_json:
        with open(opts.sys_json) as json_file:
            json_config = json.loads(jsmin(json_file.read()))
        IngestConfiguration.INGEST_CONFIG = json_config['Ingests']

    if opts.list:
        if ingest_label is None:
            list_ingest_labels(log)
            exit(0)
        elif ingest_label not in IngestConfiguration.available_ingest_labels():
            log.error("Label '%s' is not a valid ingest configuration.",
                      ingest_label)
            list_ingest_labels(log)
            exit(0)
        else:
            # List available FeatureDescriptor and Indexer configurations
            # available for the given ingest config label.
            ingest_config = IngestConfiguration(ingest_label)
            list_available_fds_idxrs(log, ingest_config)
            exit(0)

    # If we weren't given an index label, or if the one given was invalid, print
    if (ingest_label is None or ingest_label not in
            IngestConfiguration.available_ingest_labels()):
        log.error("Invalid ingest configuration specified for use: %s",
                  ingest_label)
        list_ingest_labels(log)
        exit(1)

    ingest_config = IngestConfiguration(ingest_label)

    log.info("Loading ingest instance...")
    #: :type: DataIngest or VideoIngest
    ingest = ingest_config.new_ingest_instance()

    log.info("Loading descriptor instance...")
    #: :type: SMQTK.content_description.FeatureDescriptor
    descriptor = ingest_config.new_descriptor_instance(fd_label)
    # Generate any model files needed by the chosen descriptor
    descriptor.PARALLEL = parallel
    descriptor.generate_model(ingest.data_list())

    # Don't do indexer model generation if a type was not provided
    if idxr_label:
        log.info("Loading indexer instance...")
        #: :type: SMQTK.Indexers.Indexer
        indexer = ingest_config.new_indexer_instance(idxr_label, fd_label)

        # It is not guaranteed that the feature computation method is doing
        # anything in parallel, but if it is, request that it perform serially
        # in order to allow multiple high-level feature computation jobs, else
        # we could be overrun with threads.
        descriptor.PARALLEL = 1
        # Using NonDaemonicPool because content_description that might to
        # parallel processing might use multiprocessing.Pool instances, too.
        # Pools don't usually allow daemonic processes, so this custom top-level
        # pool allows worker processes to spawn pools themselves.
        fmap = descriptor.compute_feature_async(*(df for _, df
                                                  in ingest.iteritems()),
                                                parallel=parallel,
                                                pool_type=NonDaemonicPool)

        indexer.generate_model(fmap, parallel=parallel)


if __name__ == "__main__":
    main()
