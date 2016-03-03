#!/usr/bin/env python
import csv
import logging

import matplotlib.pyplot as plt
import numpy
import sklearn.metrics

from smqtk.algorithms import (
    get_classifier_impls,
    SupervisedClassifier,
)
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


def default_config():
    return {
        'plugins': {
            'classifier':
                plugin.make_config(get_classifier_impls()),
            'classification_factory':
                ClassificationElementFactory.get_default_config(),
            'descriptor_index':
                plugin.make_config(get_descriptor_index_impls())
        },
        'utility': {
            'train': False,
            'csv_filepath': 'CHAMGEME :: PATH :: a csv file',
            'output_plot_pr': None,
            'output_plot_roc': None,
        },
        "parallelism": {
            "descriptor_fetch_cores": 4,
            "classification_cores": None,
        },
    }


def main():
    description = """
    Utility for validating a given classifier implementation's model against
    some labeled testing data, outputting PR and ROC curve plots with
    area-under-curve score values.

    This utility can optionally be used train a supervised classifier model if
    the given classifier model configuration does not exist and a second CSV
    file listing labeled training data is provided. Training will be attempted
    if ``train`` is set to true. If training is performed, we exit after
    training completes. A ``SupervisedClassifier`` sub-classing implementation
    must be configured

    We expect the test and train CSV files in the column format:

        ...
        <UUID>,<label>
        ...

    The UUID is of the descriptor to which the label applies. The label may be
    any arbitrary string value, but all labels must be consistent in
    application.
    """
    args, config = bin_utils.utility_main_helper(default_config, description)
    log = logging.getLogger(__name__)

    #
    # Initialize stuff from configuration
    #
    #: :type: smqtk.algorithms.Classifier
    classifier = plugin.from_plugin_config(
        config['plugins']['classifier'],
        get_classifier_impls()
    )
    #: :type: ClassificationElementFactory
    classification_factory = ClassificationElementFactory.from_config(
        config['plugins']['classification_factory']
    )
    #: :type: smqtk.representation.DescriptorIndex
    descriptor_index = plugin.from_plugin_config(
        config['plugins']['descriptor_index'],
        get_descriptor_index_impls()
    )

    uuid2label_filepath = config['utility']['csv_filepath']
    do_train = config['utility']['train']
    plot_filepath_pr = config['utility']['output_plot_pr']
    plot_filepath_roc = config['utility']['output_plot_roc']

    #
    # Construct mapping of label to the DescriptorElement instances for that
    # described by that label.
    #
    log.info("Loading descriptors by UUID")

    def iter_uuid_label():
        """ Iterate through UUIDs in specified file """
        with open(uuid2label_filepath) as uuid2label_file:
            reader = csv.reader(uuid2label_file)
            for r in reader:
                # TODO: This will need to be updated to handle multiple labels
                #       per descriptor.
                yield r[0], r[1]

    def get_descr(r):
        """ Fetch descriptors from configured index """
        uuid, label = r
        return label, descriptor_index.get_descriptor(uuid)

    label_element_iter = parallel.parallel_map(
        get_descr, iter_uuid_label(),
        name="cmv_get_descriptors",
        use_multiprocessing=True,
        cores=config['parallelism']['descriptor_fetch_cores'],
    )

    #: :type: dict[str, list[smqtk.representation.DescriptorElement]]
    label2descriptors = {}
    for label, d in label_element_iter:
        label2descriptors.setdefault(label, []).append(d)

    # Train classifier if the one given has a ``train`` method and training
    # was turned enabled.
    if do_train:
        if isinstance(classifier, SupervisedClassifier):
            log.info("Training classifier model")
            classifier.train(label2descriptors)
            exit(0)
        else:
            ValueError("Configured classifier is not a SupervisedClassifier "
                       "type and does not support training.")

    #
    # Apply classifier to descriptors for predictions
    #
    #: :type: dict[str, set[smqtk.representation.ClassificationElement]]
    label2classifications = {}
    for label, descriptors in label2descriptors.iteritems():
        label2classifications[label] = \
            set(classifier.classify_async(
                descriptors, classification_factory,
                use_multiprocessing=True,
                procs=config['parallelism']['classification_cores'],
                ri=1.0,
            ).values())

    #
    # Create PR/ROC curves via scikit learn tools
    #
    if plot_filepath_pr:
        log.info("Making PR curve")
        make_pr_curves(label2classifications, plot_filepath_pr)
    if plot_filepath_roc:
        log.info("Making ROC curve")
        make_roc_curves(label2classifications, plot_filepath_roc)


def format_plt(title, x_label, y_label):
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')


def make_curve(log, skl_curve_func, title, xlabel, ylabel, output_filepath,
               label2classifications):
    """
    :param skl_curve_func: scikit-learn curve generation function. This should
        be wrapped to return (x, y) value arrays of the curve plot.
    :type skl_curve_func: (list[float], list[float]) -> (list[float], list[float])

    :param label2classifications: Mapping of label to the classification
        elements that should be that label.
    :type label2classifications:
        dict[str, set[smqtk.representation.ClassificationElement]]

    """
    # Create curves for each label and then for overall.

    all_classifications = set()
    for s in label2classifications.values():
        all_classifications.update(s)

    # collection of all binary truth-probability pairs
    g_truth = []
    g_proba = []

    plt.clf()
    for l in label2classifications:
        # record binary truth relation with respect to label `l`
        l_truth = []
        l_proba = []
        for c in all_classifications:
            l_truth.append(int(c in label2classifications[l]))
            l_proba.append(c[l])
        assert len(l_truth) == len(l_proba) == len(all_classifications)
        x, y = skl_curve_func(numpy.array(l_truth),
                              numpy.array(l_proba))
        auc = sklearn.metrics.auc(x, y)
        plt.plot(x, y, label="%s (auc=%f)" % (l, auc))

        g_truth.extend(l_truth)
        g_proba.extend(l_proba)

    x, y = skl_curve_func(numpy.array(g_truth),
                          numpy.array(g_proba))
    auc = sklearn.metrics.auc(x, y)
    plt.plot(x, y, label="Overall (auc=%f)" % auc)

    format_plt(title, xlabel, ylabel)
    plt.savefig(output_filepath)


def make_pr_curves(label2classifications, output_filepath):
    def skl_pr_curve(truth, proba):
        p, r, _ = sklearn.metrics.precision_recall_curve(truth, proba,
                                                         pos_label=1)
        return r, p

    log = logging.getLogger(__name__)
    make_curve(log, skl_pr_curve, "PR", "Recall", "Precision", output_filepath,
               label2classifications)


def make_roc_curves(label2classifications, output_filepath):
    def skl_roc_curve(truth, proba):
        fpr, tpr, _ = sklearn.metrics.roc_curve(truth, proba, pos_label=1)
        return fpr, tpr

    log = logging.getLogger(__name__)
    make_curve(log, skl_roc_curve, "ROC", "False Positive Rate",
               "True Positive Rate", output_filepath, label2classifications)


if __name__ == '__main__':
    main()
