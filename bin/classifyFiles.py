#!/usr/bin/env python
"""
Based on an input, trained classifier configuration, classify a number of media
files, whose descriptor is computed by the configured descriptor generator.
Input files that classify as the given label are then output to standard out.
Thus, this script acts like a filter.
"""
import argparse
import glob
import json
import logging
import os

from smqtk.algorithms import get_classifier_impls
from smqtk.algorithms import get_descriptor_generator_impls

from smqtk.representation import ClassificationElementFactory
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.file_element import DataFileElement

from smqtk.utils import plugin
from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.bin_utils import output_config


__author__ = "paul.tunison@kitware.com"


def get_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        help='Path to the configuration file to use (JSON).')
    parser.add_argument('-g', '--generate-config',
                        default=False,
                        help='Optional file path to output a generated '
                             'configuration file to. If a configuration file '
                             'was provided, its contents will be included in '
                             'the generated output.')
    parser.add_argument('--overwrite',
                        action='store_true', default=False,
                        help='When generating a configuration file, overwrite '
                             'an existing file.')
    parser.add_argument('-d', '--debug',
                        action='store_true', default=False,
                        help='Output debug messages')

    parser.add_argument('-l', '--label',
                        type=str, default=None,
                        help='The class to filter by. This is based on the '
                             'classifier configuration/model used. If this is '
                             'not provided, we will list the available labels '
                             'in the provided classifier configuration.')
    parser.add_argument("file_globs",
                        nargs='*',
                        help='Series of shell globs specifying the files to '
                             'classify.')

    return parser


def get_default_config():
    return {
        "descriptor_factory":
            DescriptorElementFactory.get_default_config(),
        "descriptor_generator":
            plugin.make_config(get_descriptor_generator_impls),
        "classification_factory":
            ClassificationElementFactory.get_default_config(),
        "classifier":
            plugin.make_config(get_classifier_impls),
    }


def main():
    log = logging.getLogger(__name__)
    parser = get_cli_parser()
    args = parser.parse_args()

    config_path = args.config
    generate_config = args.generate_config
    config_overwrite = args.overwrite
    is_debug = args.debug

    label = args.label
    file_globs = args.file_globs

    initialize_logging(logging.getLogger(__name__),
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
        plugin.from_plugin_config(config['classifier'],
                                  get_classifier_impls)

    def log_avaialable_labels():
        log.info("Available classifier labels:")
        for l in classifier.get_labels():
            log.info("- %s", l)

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
                uuid2filepath = fp
    if not data_elements:
        raise RuntimeError("No files provided for classification.")

    log.info("Computing descriptors")
    descriptor_factory = \
        DescriptorElementFactory.from_config(config['descriptor_factory'])
    #: :type: smqtk.algorithms.DescriptorGenerator
    descriptor_generator = \
        plugin.from_plugin_config(config['descriptor_generator'],
                                  get_descriptor_generator_impls)
    descr_map = descriptor_generator\
        .compute_descriptor_async(data_elements, descriptor_factory)

    log.info("Classifying descriptors")
    classification_factory = ClassificationElementFactory \
        .from_config(config['classification_factory'])
    classification_map = classifier\
        .classify_async(descr_map.values(), classification_factory)

    log.info("Printing input file paths that classified as the given label.")
    # map of UUID to filepath:
    uuid2c = dict((c.uuid, c) for c in classification_map.itervalues())
    for data in data_elements:
        if uuid2c[data.uuid()].max_label() == label:
            print uuid2filepath[data.uuid()]


if __name__ == '__main__':
    main()
