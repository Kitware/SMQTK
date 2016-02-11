#!/usr/bin/env python
"""
K-Fold cross validate a supervised classifier, producing an ROC curve image
output.
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
from smqtk.utils import (
    bin_utils,
    file_utils,
    parallel,
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
            "classification_factory":
                ClassificationElementFactory.get_default_config(),
        },
        "cross_validation": {
            "truth_labels": None,
            "num_folds": 6,
            "random_seed": None,
            "classification_use_multiprocessing": True,
        },
        "pr_curve": {
            "enabled": True,
            "show": False,
            "plot_output_directory": None,
            "plot_file_prefix": None,
        },
    }


def classifier_kfold_validation():
    description = """
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

    - pr_curve
        - enabled
            If Precision/Recall plots should be generated.

        - show
            If we should attempt to show the graph after it has been generated
            (matplotlib).

        - plot_output_directory
            Directory to save generated plots to. If None, we will not save
            plots.

        - plot_file_prefix
            String prefix to prepend to standard plot file names.
    """
    args, config = bin_utils.utility_main_helper(default_config(), description)
    log = logging.getLogger(__name__)

    #
    # Load configurations / Setup data
    #
    use_mp = config['cross_validation']['classification_use_multiprocessing']

    pr_enabled = config['pr_curve']['enabled']
    pr_output_dir = config['pr_curve']['plot_output_directory']
    pr_file_prefix = config['pr_curve']['plot_file_prefix'] or ''
    pr_show = config['pr_curve']['show']

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
    classification_factory = ClassificationElementFactory.from_config(
        config['plugins']['classification_factory']
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
        negatives = pos_map['negative']
        del pos_map['negative']

        log.info("-- Training classifier")
        classifier.train(pos_map, negatives)

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

        # DEBUG
        if i==2: break

    #
    # PR Curve generation
    #
    if pr_enabled:
        make_pr_curves(fold_data, pr_output_dir, pr_file_prefix, pr_show)


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


def make_pr_curves(fold_data, output_dir, plot_prefix, show):
    log = logging.getLogger(__name__)
    file_utils.safe_create_dir(output_dir)

    log.info("Generating PR curves for per-folds and overall")
    # in-order list of fold (recall, precision) value lists
    fold_pr = []
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
            p, r, _ = sklearn.metrics.precision_recall_curve(l_truth, l_proba)
            auc = sklearn.metrics.auc(r, p)
            plt.plot(r, p, label="class '%s' (auc=%f)" % (label, auc))

            f_truth.extend(l_truth)
            f_proba.extend(l_proba)

        # Plot for fold
        p, r, _ = sklearn.metrics.precision_recall_curve(f_truth, f_proba)
        auc = sklearn.metrics.auc(r, p)
        plt.plot(r, p, label="Fold (auc=%f)" % auc)

        format_plt("Classifier PR - Fold %d" % i, "Recall", "Precision")
        filename = plot_prefix + 'pr.fold_%d.png' % i
        save_plt(output_dir, filename, show)

        fold_pr.append([r, p])
        fold_auc.append(auc)
        g_truth.extend(f_truth)
        g_proba.extend(f_proba)

    # Plot global curve
    log.info("-- All folds")
    plt.clf()
    for i in fold_data:
        plt.plot(fold_pr[i][0], fold_pr[i][1],
                 label="Fold %d (auc=%f)" % (i, fold_auc[i]))

    p, r, _ = sklearn.metrics.precision_recall_curve(g_truth, g_proba)
    auc = sklearn.metrics.auc(r, p)
    plt.plot(r, p, label="All (auc=%f)" % auc)

    format_plt("classifier PR - All", "Recall", "Precision")
    filename = plot_prefix + "pr.all_validataion.png"
    save_plt(output_dir, filename, show)


if __name__ == '__main__':
    classifier_kfold_validation()
