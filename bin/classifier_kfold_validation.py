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
    plot_output_dir = config['pr_curve']['plot_output_directory']
    plot_file_prefix = config['pr_curve']['plot_file_prefix'] or ''
    plot_show = config['pr_curve']['show']

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
    uuids = []
    truth_labels = []
    with open(config['cross_validation']['truth_labels']) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            uuids.append(row[0])
            truth_labels.append(row[1])
    uuids = numpy.array(uuids)
    truth_labels = numpy.array(truth_labels)

    #
    # Cross validation
    #
    kfolds = sklearn.cross_validation.StratifiedKFold(
        truth_labels, config['cross_validation']['num_folds'],
        random_state=config['cross_validation']['random_seed']
    )

    # Accumulation of binary true/false labels for each class for each fold
    global_y_true = []
    global_probs = []

    i = 0
    for train, test in kfolds:
        log.info("Fold %d", i)
        log.info("-- %d training examples", len(train))
        log.info("-- %d test examples", len(test))

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
        def iter_test_descrs():
            for idx in test:
                yield descriptor_index.get_descriptor(uuids[idx])
        m = classifier.classify_async(iter_test_descrs(),
                                      classification_factory,
                                      use_multiprocessing=use_mp,
                                      ri=1.0)
        uuid2c = dict((d.uuid(), c.get_classification())
                      for d, c in m.iteritems())

        plt.clf()

        log.info("-- Compute PR curve for each non-negative label")
        fold_y_true = []
        fold_probs = []
        # Only considering positive labels
        for t_label in pos_map:
            y_true = [l == t_label for l in truth_labels[test]]
            fold_y_true.extend(y_true)
            global_y_true.extend(y_true)

            probs = [uuid2c[uuid][t_label] for uuid in uuids[test]]
            fold_probs.extend(probs)
            global_probs.extend(probs)

            p, r, _ = sklearn.metrics.precision_recall_curve(
                y_true, probs
            )
            auc = sklearn.metrics.auc(r, p)
            plt.plot(r, p, label="Class '%s' - AUC=%f" % (t_label, auc))

        p, r, _ = sklearn.metrics.precision_recall_curve(
            fold_y_true, fold_probs
        )
        auc = sklearn.metrics.auc(r, p)
        plt.plot(r, p, 'k--', label="All Classes (AUC=%f)" % auc)

        plt.xlim([0., 1.])
        plt.ylim([0., 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Classifier PR - Fold %d" % i)
        plt.legend(loc='best')
        if plot_output_dir is not None:
            log.info("-- writing fold %d plot", i)
            of = os.path.join(plot_output_dir,
                              plot_file_prefix + 'pr.fold-%d.png' % i)
            plt.savefig(of)
        if plot_show:
            plt.show()

        i += 1

    log.info("Creating global PR curve")
    p, r, _ = sklearn.metrics.precision_recall_curve(
        global_y_true, global_probs
    )
    auc = sklearn.metrics.auc(r, p)

    plt.clf()
    plt.plot(r, p, 'k--', label='Global Cross-validation PR (AUC=%f)' % auc)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Classification PR - Global")
    plt.legend(loc='best')
    if plot_output_dir is not None:
        log.info("-- writing global plot")
        of = os.path.join(plot_output_dir,
                          plot_file_prefix + 'pr.global.png')
        plt.savefig(of)
    if plot_show:
        plt.show()


# TODO: Implement in a parallel manner using smqtk...parallel_map


if __name__ == '__main__':
    classifier_kfold_validation()
