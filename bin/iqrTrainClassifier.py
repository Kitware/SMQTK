#!/usr/bin/env python
"""
Train a supervised classifier based on an IQR session state dump.

Descriptors used in IQR, and thus referenced via their UUIDs in the IQR session
state dump, must exist external to the IQR web-app (uses a non-memory backend).
This is needed so that this script might access them for classifier training.

Getting an IQR Session's State Info
===================================
After working with the IQR application, when it is believed that the return
results are being ranked accurately, add ``iqr_session_info`` to the end of
the current URL. This will return JSON data that should be saved, and used as
input to this utility. This details the data/descriptor UUIDs that were marked
as positive and negative that will be used to train the configured classifier.

"""
import argparse
import json
import logging
import os

from smqtk.algorithms import SupervisedClassifier
from smqtk.algorithms import get_classifier_impls

from smqtk.representation import DescriptorElementFactory

from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.bin_utils import output_config
from smqtk.utils.plugin import make_config
from smqtk.utils.plugin import from_plugin_config


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
    parser.add_argument('-o', '--overwrite',
                        action='store_true', default=False,
                        help='When generating a configuration file, overwrite '
                             'an existing file.')

    parser.add_argument('-i', '--iqr-session-info',
                        help="Path to the JSON file saved from an IQR session.")
    parser.add_argument('-d', '--debug',
                        action='store_true', default=False,
                        help='Output debug messages')

    return parser


def get_default_config():
    return {
        "descriptor_element_factory":
            DescriptorElementFactory.get_default_config(),
        "classifier": make_config(get_classifier_impls),
    }


def train_classifier_iqr(config, iqrs_info):
    log = logging.getLogger(__name__)

    factory = DescriptorElementFactory\
        .from_config(config['descriptor_element_factory'])
    #: :type: smqtk.algorithms.SupervisedClassifier
    classifier = from_plugin_config(config['classifier'], get_classifier_impls)

    if not isinstance(classifier, SupervisedClassifier):
        raise RuntimeError("Configured classifier must be of the "
                           "SupervisedClassifier type in order to train.")

    log.info("Loading pos/neg descriptors")
    d_type_str = iqrs_info['descriptor_type']
    log.info("-- descriptor algo type: %s", d_type_str)
    # Using sets to handle possible descriptor duplication in example and
    # neighbor lists.
    #: :type: set[smqtk.representation.DescriptorElement]
    pos = set(
        [factory(d_type_str, uuid) for uuid in iqrs_info['positive_uids']] +
        [factory(d_type_str, uuid) for uuid in iqrs_info['ex_pos']]
    )
    #: :type: set[smqtk.representation.DescriptorElement]
    neg = set(
        [factory(d_type_str, uuid) for uuid in iqrs_info['negative_uids']] +
        [factory(d_type_str, uuid) for uuid in iqrs_info['ex_neg']]
    )

    log.info("Checking that descriptors have values")
    assert all(d.has_vector() for d in pos), \
        "Some descriptors in positive set do not have vector values."
    assert all(d.has_vector() for d in neg), \
        "Some descriptors in negative set do not have vector values."

    classifier.train({'positive': pos}, negatives=neg)


def main():
    log = logging.getLogger(__name__)
    parser = get_cli_parser()
    args = parser.parse_args()

    config_path = args.config
    generate_config = args.generate_config
    config_overwrite = args.overwrite
    iqr_session_info_fp = args.iqr_session_info
    is_debug = args.debug

    initialize_logging(logging.getLogger(),
                       is_debug and logging.DEBUG or logging.INFO)
    log.debug("Showing debug messages.")

    config = get_default_config()
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            log.info("Loading configuration: %s", config_path)
            config.update(
                json.load(f)
            )
    output_config(generate_config, config, log, config_overwrite, 100)

    if not os.path.isfile(iqr_session_info_fp):
        log.error("IQR Session info JSON filepath was invalid")
        exit(101)

    with open(iqr_session_info_fp) as f:
        iqrs_info = json.load(f)

    train_classifier_iqr(config, iqrs_info)


if __name__ == "__main__":
    main()
