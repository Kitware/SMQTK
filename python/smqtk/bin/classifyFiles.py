"""
Based on an input, trained classifier configuration, classify a number of media
files, whose descriptor is computed by the configured descriptor generator.
Input files that classify as the given label are then output to standard out.
Thus, this script acts like a filter.
"""
from __future__ import print_function

import glob
import json
import logging
import os

from smqtk.algorithms import Classifier
from smqtk.algorithms import DescriptorGenerator

from smqtk.representation import ClassificationElementFactory
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.file_element import DataFileElement

from smqtk.utils.cli import (
    initialize_logging,
    output_config,
    basic_cli_parser,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


def get_cli_parser():
    parser = basic_cli_parser(__doc__)

    g_classifier = parser.add_argument_group("Classification")
    g_classifier.add_argument('--overwrite',
                              action='store_true', default=False,
                              help='When generating a configuration file, '
                                   'overwrite an existing file.')
    g_classifier.add_argument('-l', '--label',
                              default=None,
                              help='The class to filter by. This is based on '
                                   'the classifier configuration/model used. '
                                   'If this is not provided, we will list the '
                                   'available labels in the provided '
                                   'classifier configuration.')

    # Positional
    parser.add_argument("file_globs",
                        nargs='*',
                        metavar='GLOB',
                        help='Series of shell globs specifying the files to '
                             'classify.')

    return parser


def get_default_config():
    return {
        "descriptor_factory":
            DescriptorElementFactory.get_default_config(),
        "descriptor_generator":
            make_default_config(DescriptorGenerator.get_impls()),
        "classification_factory":
            ClassificationElementFactory.get_default_config(),
        "classifier":
            make_default_config(Classifier.get_impls()),
    }


def main():
    log = logging.getLogger(__name__)
    parser = get_cli_parser()
    args = parser.parse_args()

    config_path = args.config
    generate_config = args.generate_config
    config_overwrite = args.overwrite
    is_debug = args.verbose

    label = args.label
    file_globs = args.file_globs

    initialize_logging(logging.getLogger('__main__'),
                       is_debug and logging.DEBUG or logging.INFO)
    initialize_logging(logging.getLogger('smqtk'),
                       is_debug and logging.DEBUG or logging.INFO)
    log.debug("Showing debug messages.")

    config = get_default_config()
    config_loaded = False
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            log.info("Loading configuration: %s", config_path)
            config.update(
                json.load(f)
            )
        config_loaded = True
    output_config(generate_config, config, log, config_overwrite, 100)

    if not config_loaded:
        log.error("No configuration provided")
        exit(101)

    classify_files(config, label, file_globs)


def classify_files(config, label, file_globs):
    log = logging.getLogger(__name__)

    #: :type: smqtk.algorithms.Classifier
    classifier = \
        from_config_dict(config['classifier'], Classifier.get_impls())

    def log_avaialable_labels():
        log.info("Available classifier labels:")
        for _l in classifier.get_labels():
            log.info("- %s", _l)

    if label is None:
        log_avaialable_labels()
        return
    elif label not in classifier.get_labels():
        log.error("Invalid classification label provided to compute and filter "
                  "on: '%s'", label)
        log_avaialable_labels()
        return

    log.info("Collecting files from globs")
    #: :type: list[DataFileElement]
    data_elements = []
    uuid2filepath = {}
    for g in file_globs:
        if os.path.isfile(g):
            d = DataFileElement(g)
            data_elements.append(d)
            uuid2filepath[d.uuid()] = g
        else:
            log.debug("expanding glob: %s", g)
            for fp in glob.iglob(g):
                d = DataFileElement(fp)
                data_elements.append(d)
                uuid2filepath[d.uuid()] = fp
    if not data_elements:
        raise RuntimeError("No files provided for classification.")

    log.info("Computing descriptors")
    descriptor_factory = \
        DescriptorElementFactory.from_config(config['descriptor_factory'])
    #: :type: smqtk.algorithms.DescriptorGenerator
    descriptor_generator = \
        from_config_dict(config['descriptor_generator'],
                         DescriptorGenerator.get_impls())
    descr_iter = descriptor_generator.generate_elements(
        data_elements, descr_factory=descriptor_factory
    )

    log.info("Classifying descriptors")
    classification_factory = ClassificationElementFactory \
        .from_config(config['classification_factory'])
    classification_iter = \
        classifier.classify_elements(descr_iter, classification_factory)

    log.info("Printing input file paths that classified as the given label.")
    # map of UUID to filepath:
    uuid2c = {c.uuid: c for c in classification_iter}
    for data in data_elements:
        d_uuid = data.uuid()
        log.debug("'{}' classification map: {}".format(
            uuid2filepath[d_uuid], uuid2c[d_uuid].get_classification()
        ))
        if uuid2c[d_uuid].max_label() == label:
            print(uuid2filepath[d_uuid])


if __name__ == '__main__':
    main()
