"""
Train a supervised classifier based on an IQR session state dump.

Descriptors used in IQR, and thus referenced via their UUIDs in the IQR session
state dump, must exist external to the IQR web-app (uses a non-memory backend).
This is needed so that this script might access them for classifier training.

Getting an IQR Session's State Information
==========================================
Click the "Save IQR State" button to download the IqrState file encapsulating
the descriptors of positively and negatively marked items. These descriptors
will be used to train the configured SupervisedClassifier.

"""
import argparse
import json
import logging
import os
import zipfile

import numpy

from smqtk.algorithms import SupervisedClassifier
from smqtk.algorithms import get_classifier_impls

from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement

from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.bin_utils import output_config
from smqtk.utils.plugin import make_config
from smqtk.utils.plugin import from_plugin_config


__author__ = "paul.tunison@kitware.com"


def get_cli_parser():
    description = """
    Train a supervised classifier based on an IQR session state dump.

    Descriptors used in IQR, and thus referenced via their UUIDs in the IQR
    session state dump, must exist external to the IQR web-app (uses a
    non-memory backend). This is needed so that this script might access them
    for classifier training.

    Getting an IQR Session's State Information
    ==========================================
    Click the "Save IQR State" button to download the IqrState file
    encapsulating the descriptors of positively and negatively marked items.
    These descriptors will be used to train the configured
    SupervisedClassifier.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)

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

    parser.add_argument('-i', '--iqr-state',
                        help="Path to the ZIP file saved from an IQR session.")
    parser.add_argument('-d', '--debug',
                        action='store_true', default=False,
                        help='Output debug messages')

    return parser


def get_default_config():
    return {
        "classifier": make_config(get_classifier_impls()),
    }


def train_classifier_iqr(config, iqr_state_fp):
    log = logging.getLogger(__name__)

    #: :type: smqtk.algorithms.SupervisedClassifier
    classifier = from_plugin_config(config['classifier'], get_classifier_impls())

    if not isinstance(classifier, SupervisedClassifier):
        raise RuntimeError("Configured classifier must be of the "
                           "SupervisedClassifier type in order to train.")

    # Get pos/neg descriptors out of iqr state zip
    z_file = open(iqr_state_fp, 'r')
    z = zipfile.ZipFile(z_file)
    if len(z.namelist()) != 1:
        raise RuntimeError("Invalid IqrState file!")
    iqrs = json.loads(z.read(z.namelist()[0]))
    if len(iqrs) != 2:
        raise RuntimeError("Invalid IqrState file!")
    if 'pos' not in iqrs or 'neg' not in iqrs:
        raise RuntimeError("Invalid IqrState file!")

    log.info("Loading pos/neg descriptors")
    #: :type: list[smqtk.representation.DescriptorElement]
    pos = []
    #: :type: list[smqtk.representation.DescriptorElement]
    neg = []
    i = 0
    for v in set(map(tuple, iqrs['pos'])):
        d = DescriptorMemoryElement('train', i)
        d.set_vector(numpy.array(v))
        pos.append(d)
        i += 1
    for v in set(map(tuple, iqrs['neg'])):
        d = DescriptorMemoryElement('train', i)
        d.set_vector(numpy.array(v))
        neg.append(d)
        i += 1
    log.info('    positive -> %d', len(pos))
    log.info('    negative -> %d', len(neg))

    classifier.train(positive=pos, negative=neg)


def main():
    log = logging.getLogger(__name__)
    parser = get_cli_parser()
    args = parser.parse_args()

    config_path = args.config
    generate_config = args.generate_config
    config_overwrite = args.overwrite
    iqr_state_fp = args.iqr_state
    is_debug = args.debug

    initialize_logging(logging.getLogger(),
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

    if not os.path.isfile(iqr_state_fp):
        log.error("IQR Session info JSON filepath was invalid")
        exit(102)

    train_classifier_iqr(config, iqr_state_fp)


if __name__ == "__main__":
    main()
