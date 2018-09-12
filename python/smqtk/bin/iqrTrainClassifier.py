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

import logging
import os

from smqtk.algorithms import SupervisedClassifier
from smqtk.algorithms import get_classifier_impls

from smqtk.iqr import IqrSession

from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement

from smqtk.utils.bin_utils import (
    basic_cli_parser,
    utility_main_helper,
)
from smqtk.utils.plugin import make_config
from smqtk.utils.plugin import from_plugin_config


def get_cli_parser():
    parser = basic_cli_parser(__doc__)
    parser.add_argument('-i', '--iqr-state',
                        help="Path to the ZIP file saved from an IQR session.")
    return parser


def get_default_config():
    return {
        "classifier": make_config(get_classifier_impls()),
    }


def train_classifier_iqr(config, iqr_state_fp):
    #: :type: smqtk.algorithms.SupervisedClassifier
    classifier = from_plugin_config(
        config['classifier'],
        get_classifier_impls(sub_interface=SupervisedClassifier)
    )

    # Load state into an empty IqrSession instance.
    with open(iqr_state_fp, 'rb') as f:
        state_bytes = f.read().strip()
    descr_factory = DescriptorElementFactory(DescriptorMemoryElement, {})
    iqrs = IqrSession()
    iqrs.set_state_bytes(state_bytes, descr_factory)

    # Positive descriptor examples for training are composed of those from
    # external and internal sets. Same for negative descriptor examples.
    pos = iqrs.positive_descriptors | iqrs.external_positive_descriptors
    neg = iqrs.negative_descriptors | iqrs.external_negative_descriptors
    classifier.train(class_examples={'positive': pos, 'negative': neg})


def main():
    args = get_cli_parser().parse_args()
    config = utility_main_helper(get_default_config, args)

    log = logging.getLogger(__name__)
    log.debug("Showing debug messages.")

    iqr_state_fp = args.iqr_state

    if not os.path.isfile(iqr_state_fp):
        log.error("IQR Session info JSON filepath was invalid")
        exit(102)

    train_classifier_iqr(config, iqr_state_fp)


if __name__ == "__main__":
    main()
