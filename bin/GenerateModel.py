#!/usr/bin/env python
"""
Generate a classifier model for the given ingest data

Results are

"""

import json
import logging
import os.path as osp

import smqtk_config

from SMQTK.Classifiers import get_classifiers
from SMQTK.FeatureDescriptors import get_descriptors
from SMQTK.utils import DataIngest, VideoIngest


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
        print
        print "Feature Descriptors:"
        for name in get_descriptors().keys():
            print "\t%s" % name
        print
        print "Classifiers:"
        for name in get_classifiers().keys():
            print "\t%s" % name
        print
        exit(0)

    descriptor_t = get_descriptors()[opts.feature_descriptor]
    classifier_t = get_classifiers()[opts.classifier]
    data_dir = opts.data_dir or smqtk_config.DATA_DIR
    work_dir = opts.work_dir or smqtk_config.WORK_DIR
    if opts.sys_json:
        with open(opts.sys_json) as ifile:
            sc = json.load(ifile)
    else:
        sc = smqtk_config.SYSTEM_CONFIG

    if opts.type == 'image':
        ingest_t = DataIngest
    elif opts.type == 'video':
        ingest_t = VideoIngest
    else:
        raise RuntimeError("Invalid ingest type! Given: %s" % opts.type)
    t = opts.type
    t = t[0].upper() + t[1:]

    ingest = ingest_t(osp.join(data_dir, sc['Ingest'][t]),
                      osp.join(work_dir, sc['Ingest'][t]))
    descriptor = descriptor_t(osp.join(data_dir,
                                       sc['FeatureDescriptors']
                                         [opts.feature_descriptor]
                                         ['data_directory']),
                              osp.join(work_dir,
                                       sc['FeatureDescriptors']
                                         [opts.feature_descriptor]
                                         ['data_directory']))
    classifier = classifier_t(osp.join(data_dir,
                                       sc["Classifiers"]
                                         [opts.classifier]
                                         [opts.feature_descriptor]
                                         ['data_directory']),
                              osp.join(work_dir,
                                       sc["Classifiers"]
                                         [opts.classifier]
                                         [opts.feature_descriptor]
                                         ['data_directory']))

    fmap = descriptor.compute_feature_async(*(df for _, df in ingest.iteritems()))
    classifier.generate_model(fmap)


if __name__ == "__main__":
    main()