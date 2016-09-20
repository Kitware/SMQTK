"""
Helper utility for cross validating a supervised classifier configuration.
The classifier used should NOT be configured to save its model since this
process requires us to train the classifier multiple times.

Configuration
-------------
    - plugins
        - supervised_classifier
            Supervised Classifier implementation configuration to use. This
            should not be set to use a persistent model if able.

        - descriptor_index
            Index to draw descriptors to classify from.

    - cross_validation
        - truth_labels
            Path to a CSV file containing descriptor UUID the truth label
            associations. This defines what descriptors are used from the given
            index. We error if any descriptor UUIDs listed here are not
            available in the given descriptor index. This file should be in
            [uuid, label] column format.

        - num_folds
            Number of folds to make for cross validation.

        - random_seed
            Optional fixed seed for the

        - classification_use_multiprocessing
            If we should use multiprocessing (vs threading) when classifying
            elements.

    - pr_curves
        - enabled
            If Precision/Recall plots should be generated.

        - show
            If we should attempt to show the graph after it has been generated
            (matplotlib).

        - output_directory
            Directory to save generated plots to. If None, we will not save
            plots. Otherwise we will create the directory (and required parent
            directories) if it does not exist.

        - file_prefix
            String prefix to prepend to standard plot file names.

    - roc_curves
        - enabled
            If ROC curves should be generated

        - show
            If we should attempt to show the plot after it has been generated
            (matplotlib).

        - output_directory
            Directory to save generated plots to. If None, we will not save
            plots. Otherwise we will create the directory (and required parent
            directories) if it does not exist.

        - file_prefix
            String prefix to prepend to standard plot file names.
"""

import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy
import sklearn.cross_validation
import sklearn.metrics

from smqtk.algorithms import get_classifier_impls
from smqtk.algorithms.classifier import SupervisedClassifier
from smqtk.representation import (
    ClassificationElementFactory,
    get_descriptor_index_impls,
)
from smqtk.representation.classification_element.memory import \
    MemoryClassificationElement
from smqtk.utils import (
    bin_utils,
    file_utils,
    plugin,
)


__author__ = "paul.tunison@kitware.com"


def get_supervised_classifier_impls():
    return get_classifier_impls(sub_interface=SupervisedClassifier)


def default_config():
    return {
        "plugins": {
            "supervised_classifier":
                plugin.make_config(get_supervised_classifier_impls()),
            "descriptor_index":
                plugin.make_config(get_descriptor_index_impls()),
        },
        "cross_validation": {
            "truth_labels": None,
            "num_folds": 6,
            "random_seed": None,
            "classification_use_multiprocessing": True,
        },
        "pr_curves": {
            "enabled": True,
            "show": False,
            "output_directory": None,
            "file_prefix": None,
        },
        "roc_curves": {
            "enabled": True,
            "show": False,
            "output_directory": None,
            "file_prefix": None,
        },
    }


def cli_parser():
    return bin_utils.basic_cli_parser(__doc__)


def classifier_kfold_validation():
    args = cli_parser().parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    #
    # Load configurations / Setup data
    #
    use_mp = config['cross_validation']['classification_use_multiprocessing']

    pr_enabled = config['pr_curves']['enabled']
    pr_output_dir = config['pr_curves']['output_directory']
    pr_file_prefix = config['pr_curves']['file_prefix'] or ''
    pr_show = config['pr_curves']['show']

    roc_enabled = config['roc_curves']['enabled']
    roc_output_dir = config['roc_curves']['output_directory']
    roc_file_prefix = config['roc_curves']['file_prefix'] or ''
    roc_show = config['roc_curves']['show']

    log.info("Initializing DescriptorIndex (%s)",
             config['plugins']['descriptor_index']['type'])
    #: :type: smqtk.representation.DescriptorIndex
    descriptor_index = plugin.from_plugin_config(
        config['plugins']['descriptor_index'],
        get_descriptor_index_impls()
    )
    log.info("Loading classifier configuration")
    #: :type: dict
    classifier_config = config['plugins']['supervised_classifier']

    # Always use in-memory ClassificationElement since we are retraining the
    # classifier and don't want possible element caching
    #: :type: ClassificationElementFactory
    classification_factory = ClassificationElementFactory(
        MemoryClassificationElement, {}
    )

    log.info("Loading truth data")
    #: :type: list[str]
    uuids = []
    #: :type: list[str]
    truth_labels = []
    with open(config['cross_validation']['truth_labels']) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            uuids.append(row[0])
            truth_labels.append(row[1])
    #: :type: numpy.ndarray[str]
    uuids = numpy.array(uuids)
    #: :type: numpy.ndarray[str]
    truth_labels = numpy.array(truth_labels)

    #
    # Cross validation
    #
    kfolds = sklearn.cross_validation.StratifiedKFold(
        truth_labels, config['cross_validation']['num_folds'],
        random_state=config['cross_validation']['random_seed']
    )

    """
    Truth and classification probability results for test data per fold.
    Format:
        {
            0: {
                '<label>':  {
                    "truth": [...],   # Parallel truth and classification
                    "proba": [...],   # probability values
                },
                ...
            },
            ...
        }
    """
    fold_data = {}

    i = 0
    for train, test in kfolds:
        log.info("Fold %d", i)
        log.info("-- %d training examples", len(train))
        log.info("-- %d test examples", len(test))
        fold_data[i] = {}

        log.info("-- creating classifier")
        #: :type: SupervisedClassifier
        classifier = plugin.from_plugin_config(
            classifier_config,
            get_supervised_classifier_impls()
        )

        log.info("-- gathering descriptors")
        #: :type: dict[str, list[smqtk.representation.DescriptorElement]]
        pos_map = {}
        for idx in train:
            if truth_labels[idx] not in pos_map:
                pos_map[truth_labels[idx]] = []
            pos_map[truth_labels[idx]].append(
                descriptor_index.get_descriptor(uuids[idx])
            )

        log.info("-- Training classifier")
        classifier.train(pos_map)

        log.info("-- Classifying test set")
        m = classifier.classify_async(
            (descriptor_index.get_descriptor(uuids[idx]) for idx in test),
            classification_factory,
            use_multiprocessing=use_mp, ri=1.0
        )
        uuid2c = dict((d.uuid(), c.get_classification())
                      for d, c in m.iteritems())

        log.info("-- Pairing truth and computed probabilities")
        # Only considering positive labels
        for t_label in pos_map:
            fold_data[i][t_label] = {
                "truth": [l == t_label for l in truth_labels[test]],
                "proba": [uuid2c[uuid][t_label] for uuid in uuids[test]]
            }

        i += 1

    #
    # Curve generation
    #
    if pr_enabled:
        make_pr_curves(fold_data, pr_output_dir, pr_file_prefix, pr_show)
    if roc_enabled:
        make_roc_curves(fold_data, roc_output_dir, roc_file_prefix, roc_show)


def format_plt(title, x_label, y_label):
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')


def save_plt(output_dir, file_name, show):
    file_utils.safe_create_dir(output_dir)
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path)
    if show:
        plt.show()


def make_curves(log, skl_curve_func, title_hook, x_label, y_label, fold_data,
                output_dir, plot_prefix, show):
    """
    Generic method for PR/ROC curve generation

    :param skl_curve_func: scikit-learn curve generation function. This should
        be wrapped to return (x, y) value arrays.
    """
    file_utils.safe_create_dir(output_dir)

    log.info("Generating %s curves for per-folds and overall", title_hook)
    # in-order list of fold (x, y) value lists
    fold_xy = []
    fold_auc = []

    # all truth and proba pairs
    g_truth = []
    g_proba = []

    for i in fold_data:
        log.info("-- Fold %i", i)
        f_truth = []
        f_proba = []

        plt.clf()
        for label in fold_data[i]:
            log.info("   -- label '%s'", label)
            l_truth = fold_data[i][label]['truth']
            l_proba = fold_data[i][label]['proba']
            x, y = skl_curve_func(l_truth, l_proba)
            auc = sklearn.metrics.auc(x, y)
            plt.plot(x, y, label="class '%s' (auc=%f)" % (label, auc))

            f_truth.extend(l_truth)
            f_proba.extend(l_proba)

        # Plot for fold
        x, y = skl_curve_func(f_truth, f_proba)
        auc = sklearn.metrics.auc(x, y)
        plt.plot(x, y, label="Fold (auc=%f)" % auc)

        format_plt("Classifier %s - Fold %d" % (title_hook, i),
                   x_label, y_label)
        filename = plot_prefix + 'fold_%d.png' % i
        save_plt(output_dir, filename, show)

        fold_xy.append([x, y])
        fold_auc.append(auc)
        g_truth.extend(f_truth)
        g_proba.extend(f_proba)

    # Plot global curve
    log.info("-- All folds")
    plt.clf()
    for i in fold_data:
        plt.plot(fold_xy[i][0], fold_xy[i][1],
                 label="Fold %d (auc=%f)" % (i, fold_auc[i]))

    x, y = skl_curve_func(g_truth, g_proba)
    auc = sklearn.metrics.auc(x, y)
    plt.plot(x, y, label="All (auc=%f)" % auc)

    format_plt("Classifier %s - Validation" % title_hook, x_label, y_label)
    filename = plot_prefix + "validation.png"
    save_plt(output_dir, filename, show)


def make_pr_curves(fold_data, output_dir, plot_prefix, show):
    log = logging.getLogger(__name__)

    def skl_pr_curve(truth, proba):
        p, r, _ = sklearn.metrics.precision_recall_curve(truth, proba)
        return r, p

    make_curves(log, skl_pr_curve, "PR", "Recall", "Precision", fold_data,
                output_dir, plot_prefix + 'pr.', show)


def make_roc_curves(fold_data, output_dir, plot_prefix, show):
    log = logging.getLogger(__name__)

    def skl_roc_curve(truth, proba):
        fpr, tpr, _ = sklearn.metrics.roc_curve(truth, proba)
        return fpr, tpr

    make_curves(log, skl_roc_curve, "ROC", "False Positive Rate",
                "True Positive Rate", fold_data, output_dir,
                plot_prefix + 'roc.', show)


if __name__ == '__main__':
    classifier_kfold_validation()
