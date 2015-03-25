#!/usr/bin/env python
"""
Generate a classifier model for the given ingest data

Results are

"""

import json
import logging
import multiprocessing.pool
import os.path as osp

import smqtk_config

from SMQTK.Indexers import get_indexers
from SMQTK.FeatureDescriptors import get_descriptors, FeatureDescriptor
from SMQTK.utils import DataIngest, VideoIngest


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
        "Generate the model for the given classifier type, using features " \
        "from the given feature descriptor type. We use configured valued in " \
        "the smqtk_config module and from the system configuration JSON file " \
        "(etc/system_config.json) unless otherwise specified by options to " \
        "this script. Specific ingest used is determined by the ingest type " \
        "provided (-t/--type)."
    parser = optparse.OptionParser(description=description)
    groupRequired = optparse.OptionGroup(parser, "Required Options")
    groupOptional = optparse.OptionGroup(parser, "Optional")

    groupRequired.add_option('-t', '--type',
                             help="Ingest data type. Currently supports "
                                  "'image' or 'video'. This determines which "
                                  "data ingest to use.")
    groupRequired.add_option('-d', '--feature-descriptor',
                             help="Type of feature descriptor to use for "
                                  "feature generation.")
    groupRequired.add_option('-c', '--classifier',
                             help="Type of classifier to generate the model "
                                  "for.")

    groupOptional.add_option('--data-dir',
                             help='Custom data directory to use. Otherwise '
                                  'we pull the data directory from the '
                                  'smqtk_config module.')
    groupOptional.add_option('--work-dir',
                             help="Custom work directory to use. Otherwise "
                                  "we pull the work directory from the "
                                  "smqtk_config module.")
    groupOptional.add_option('--sys-json',
                             help="Custom system configuration JSON file to "
                                  "use. Otherwise we use the one specified in "
                                  "the smqtk_config module.")
    groupOptional.add_option('-l', '--list', action='store_true', default=False,
                             help="List available FeatureDescriptor and "
                                  "Classifier types.")
    groupOptional.add_option('-v', '--verbose', action='store_true',
                             default=False,
                             help='Add debug messaged to output logging.')

    parser.add_option_group(groupRequired)
    parser.add_option_group(groupOptional)
    opts, args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    if opts.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if opts.list:
        fds = get_descriptors().keys()
        cls = get_indexers().keys()

        print
        print "Feature Descriptors:"
        print
        for name in fds:
            print "\t%s" % name
        print
        print
        print "Indexers:"
        print
        for name in cls:
            print "\t%s" % name
        print
        exit(0)

    descriptor_t = get_descriptors()[opts.feature_descriptor]
    classifier_t = get_indexers()[opts.classifier]

    abspath = lambda p: osp.abspath(osp.expanduser(p))
    data_dir = abspath(opts.data_dir or smqtk_config.DATA_DIR)
    work_dir = abspath(opts.work_dir or smqtk_config.WORK_DIR)

    if opts.sys_json:
        with open(abspath(opts.sys_json)) as ifile:
            sc = json.load(ifile)
    else:
        sc = smqtk_config.SYSTEM_CONFIG

    if opts.type.lower() == 'image':
        ingest_t = DataIngest
    elif opts.type.lower() == 'video':
        ingest_t = VideoIngest
    else:
        raise RuntimeError("Invalid ingest type! Given: %s" % opts.type)
    t = opts.type.lower()
    t = t[0].upper() + t[1:]

    #: :type: DataIngest or VideoIngest
    ingest = ingest_t(osp.join(data_dir, sc['Ingest'][t]),
                      osp.join(work_dir, sc['Ingest'][t]))
    #: :type: SMQTK.FeatureDescriptors.FeatureDescriptor
    descriptor = descriptor_t(osp.join(data_dir,
                                       sc['FeatureDescriptors']
                                         [opts.feature_descriptor]
                                         ['data_directory']),
                              osp.join(work_dir,
                                       sc['FeatureDescriptors']
                                         [opts.feature_descriptor]
                                         ['data_directory']))
    #: :type: SMQTK.Indexers.Indexer
    classifier = classifier_t(osp.join(data_dir,
                                       sc["Indexers"]
                                         [opts.classifier]
                                         [opts.feature_descriptor]
                                         ['data_directory']),
                              osp.join(work_dir,
                                       sc["Indexers"]
                                         [opts.classifier]
                                         [opts.feature_descriptor]
                                         ['data_directory']))

    # Generate any model files needed by the chosen descriptor
    descriptor.generate_model(ingest.data_list())

    # It is not guaranteed that the feature computation method is doing anything
    # in parallel, but if it is, request that it perform serially in order to
    # allow multiple high-level feature computation jobs.
    FeatureDescriptor.PARALLEL = 1
    # Using NonDaemonicPool because FeatureDescriptors that might to parallel
    # processing might use multiprocessing.Pool instances, too. Pools don't
    # usually allow daemonic processes, so this custom top-level pool allows
    # worker processes to spawn pools themselves.
    fmap = descriptor.compute_feature_async(*(df for _, df in ingest.iteritems()),
                                            pool_type=NonDaemonicPool)
    classifier.generate_model(fmap)


if __name__ == "__main__":
    main()
