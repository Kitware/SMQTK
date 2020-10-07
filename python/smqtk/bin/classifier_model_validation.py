"""
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

Some metrics presented assume the highest confidence class as the single
predicted class for an element:

    - confusion matrix

The output UUID confusion matrix is a JSON dictionary where the top-level
keys are the true labels, and the inner dictionary is the mapping of
predicted labels to the UUIDs of the classifications/descriptors that
yielded the prediction. Again, this is based on the maximum probability
label for a classification result (T=0.5).

See Scikit-Learn PR and ROC curve explanations and examples:
    - http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    - http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

"""
from __future__ import division, print_function

import collections
import csv
import json
import logging
from typing import Dict, Hashable, List, Union
import warnings

import matplotlib.pyplot as plt  # type: ignore
import numpy
import scipy.stats
import six
from six.moves import range
import sklearn.metrics

from smqtk.algorithms import (
    SupervisedClassifier,
)
from smqtk.representation import (
    ClassificationElementFactory,
    DescriptorElement,
    DescriptorSet,
)
from smqtk.utils import (
    cli,
    parallel,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


__author__ = "paul.tunison@kitware.com"


def default_config():
    return {
        'plugins': {
            'classifier':
                make_default_config(SupervisedClassifier.get_impls()),
            'classification_factory':
                ClassificationElementFactory.get_default_config(),
            'descriptor_set':
                make_default_config(DescriptorSet.get_impls())
        },
        'utility': {
            'train': False,
            'csv_filepath': 'CHAMGEME :: PATH :: a csv file',
            'output_plot_pr': None,
            'output_plot_roc': None,
            'output_plot_confusion_matrix': None,
            'output_uuid_confusion_matrix': None,
            'curve_confidence_interval': False,
            'curve_confidence_interval_alpha': 0.4,
        },
        "parallelism": {
            "descriptor_fetch_cores": 4,
            # DEPRECATED
            "classification_cores": None,
        },
    }


def cli_parser():
    return cli.basic_cli_parser(__doc__)


def main():
    args = cli_parser().parse_args()
    config = cli.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    # Deprecations
    if (config.get('parallelism', {})
              .get('classification_cores', None) is not None):
        warnings.warn("Usage of 'classification_cores' is deprecated. "
                      "Classifier parallelism is not defined on a "
                      "per-implementation basis. See classifier "
                      "implementation parameterization.",
                      category=DeprecationWarning)

    #
    # Initialize stuff from configuration
    #
    #: :type: smqtk.algorithms.Classifier
    classifier = from_config_dict(
        config['plugins']['classifier'],
        SupervisedClassifier.get_impls()
    )
    #: :type: ClassificationElementFactory
    classification_factory = ClassificationElementFactory.from_config(
        config['plugins']['classification_factory']
    )
    #: :type: smqtk.representation.DescriptorSet
    descriptor_set = from_config_dict(
        config['plugins']['descriptor_set'],
        DescriptorSet.get_impls()
    )

    uuid2label_filepath = config['utility']['csv_filepath']
    do_train = config['utility']['train']
    output_uuid_cm = config['utility']['output_uuid_confusion_matrix']
    plot_filepath_pr = config['utility']['output_plot_pr']
    plot_filepath_roc = config['utility']['output_plot_roc']
    plot_filepath_cm = config['utility']['output_plot_confusion_matrix']
    plot_ci = config['utility']['curve_confidence_interval']
    plot_ci_alpha = config['utility']['curve_confidence_interval_alpha']

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
        uuid, truth_label = r
        return truth_label, descriptor_set.get_descriptor(uuid)

    tlabel_element_iter = parallel.parallel_map(
        get_descr, iter_uuid_label(),
        name="cmv_get_descriptors",
        use_multiprocessing=True,
        cores=config['parallelism']['descriptor_fetch_cores'],
    )

    # Map of truth labels to descriptors of labeled data
    tlabel2descriptors: Dict[str, List[DescriptorElement]] = {}
    for tlabel, d in tlabel_element_iter:
        tlabel2descriptors.setdefault(tlabel, []).append(d)

    # Train classifier if the one given has a ``train`` method and training
    # was turned enabled.
    if do_train:
        log.info("Training supervised classifier model")
        classifier.train(tlabel2descriptors)
        exit(0)

    #
    # Apply classifier to descriptors for predictions
    #

    # Truth label to predicted classification results
    #: :type: dict[str, set[smqtk.representation.ClassificationElement]]
    tlabel2classifications = {}
    for tlabel, descriptors in six.iteritems(tlabel2descriptors):
        tlabel2classifications[tlabel] = \
            set(classifier.classify_elements(descriptors,
                                             classification_factory))
    log.info("Truth label counts:")
    for tlabel in sorted(tlabel2classifications):
        log.info("  %s :: %d", tlabel, len(tlabel2classifications[tlabel]))

    #
    # Confusion Matrix
    #
    conf_mat, labels = gen_confusion_matrix(tlabel2classifications)
    log.info("Confusion_matrix")
    log_cm(log.info, conf_mat, labels)
    if plot_filepath_cm:
        plot_cm(conf_mat, labels, plot_filepath_cm)

    # Confusion Matrix of descriptor UUIDs to output json
    if output_uuid_cm:
        # Top dictionary keys are true labels, inner dictionary keys are UUID
        # predicted labels.
        log.info("Computing UUID Confusion Matrix")
        uuid_cm: Dict[str, Dict[Hashable, Union[List, List]]] = {}
        for tlabel in tlabel2classifications:
            tlabel_uuid_cm = collections.defaultdict(set)
            for c in tlabel2classifications[tlabel]:
                tlabel_uuid_cm[c.max_label()].add(c.uuid)
            # convert sets to lists for master JSON output.
            uuid_cm[tlabel] = {}
            for plabel in tlabel_uuid_cm:
                uuid_cm[tlabel][plabel] = list(tlabel_uuid_cm[plabel])
        with open(output_uuid_cm, 'w') as f:
            log.info("Saving UUID Confusion Matrix: %s", output_uuid_cm)
            json.dump(uuid_cm, f, indent=2, separators=(',', ': '))

    #
    # Create PR/ROC curves via scikit learn tools
    #
    if plot_filepath_pr:
        log.info("Making PR curve")
        make_pr_curves(tlabel2classifications, plot_filepath_pr,
                       plot_ci, plot_ci_alpha)
    if plot_filepath_roc:
        log.info("Making ROC curve")
        make_roc_curves(tlabel2classifications, plot_filepath_roc,
                        plot_ci, plot_ci_alpha)


def gen_confusion_matrix(tlabel2classifications):
    """
    Generate numpy confusion matrix based on classification highest confidence
    score.

    :param tlabel2classifications: Mapping of true label for mapped set of
        classifications.
    :type tlabel2classifications:
        dict[str, set[smqtk.representation.ClassificationElement]]

    :return: Numpy confusion matrix and label vectors for rows and columns
    :rtype: numpy.ndarray[int], list[str]

    """
    # List of true and predicted classes for classifications
    true_classes = []
    pred_classes = []

    for true_label in tlabel2classifications:
        for c in tlabel2classifications[true_label]:
            true_classes.append(true_label)
            # Assuming classifier labels are strings (true labels are strings).
            pred_classes.append(str(c.max_label()))

    labels = sorted(set(true_classes).union(pred_classes))
    confusion_mat = sklearn.metrics.confusion_matrix(true_classes,
                                                     pred_classes,
                                                     labels)

    return confusion_mat, labels


def log_cm(p_func, conf_mat, labels):
    print_mat = numpy.zeros((conf_mat.shape[0] + 2, conf_mat.shape[1] + 1),
                            dtype=object)
    print_mat[0, 0] = "Predicted"
    print_mat[1, 0] = "Actual"
    print_mat[2:, 1:] = conf_mat.astype(str)
    print_mat[0, 1:] = labels
    print_mat[1, 1:] = ''
    print_mat[2:, 0] = labels

    # get col max widths
    col_max_lens = []
    for x in range(print_mat.shape[1]):
        col_max_lens.append(max(list(
            map(len, print_mat[:, x].flatten().tolist())
        )))

    # Construct printed rows based on column max width
    p_func("Confusion Matrix (Counts)")
    for r in print_mat:
        segs = []
        for i, w in enumerate(r):
            segs.append(' ' * (col_max_lens[i] - len(w)) + w)
        p_func(' '.join(segs))


def plot_cm(conf_mat, labels, output_path):
    """
    :param conf_mat: Confusion matrix with items counts
    :type conf_mat: numpy.ndarray

    :param labels: Symmetric row/column labels
    :type labels: list[str]

    :param output_path: Path to save generated figure.
    :type output_path: str

    :return:
    :rtype:

    """
    log = logging.getLogger(__name__)

    #: :type: numpy.ndarray
    cm = conf_mat.copy()
    log.debug("raw conf mat:\n%s", cm)

    # each row represents a true class
    cm_f = cm / cm.sum(1).astype(float)[:, numpy.newaxis]
    log.debug("normalized conf mat:\n%s", cm_f)

    fig = plt.figure()
    #: :type: matplotlib.axes._axes.Axes
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_f, vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    # Annotate cells with count values
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            ax.annotate(s=str(cm[y, x]), xy=(x, y), xycoords='data')

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set_title('Confusion Matrix - Percent Makeup')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    fig.savefig(output_path)


def format_plt(title, x_label, y_label):
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)


def select_color_marker(i):
    """ Return index-based marker/color format for plotting """
    colors = ['b', 'g', 'r', 'c', 'y', 'k']
    style = ['-', '--', '-.', ':']
    ci = i % len(colors)
    si = (i // len(colors)) % len(style)
    return '%s%s' % (colors[ci], style[si])


def make_curve(log, skl_curve_func, title, xlabel, ylabel, output_filepath,
               label2classifications, plot_ci, plot_ci_alpha,
               plot_ci_make_poly):
    """
    :param log: Logger to use.
    :type log: logging.Logger

    :param skl_curve_func: scikit-learn curve generation function. This should
        be wrapped to return (x, y) value arrays of the curve plot.
    :type skl_curve_func:
        (numpy.ndarray[float], numpy.ndarray[float]) ->
            (numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float])

    :param title: Title of the plot.
    :param xlabel: X-axis label for the plot.
    :param ylabel: Y-axis label for the plot.
    :param output_filepath: Path to write the generated plot image to.

    :param label2classifications: Mapping of label to the classification
        elements that should be that label.
    :type label2classifications:
        dict[str, set[smqtk.representation.ClassificationElement]]

    :param plot_ci: Flag for whether to draw the confidence interval or not.
    :type plot_ci: bool

    :param plot_ci_alpha: Alpha value to use for coloring the confidence
        interval area in the range [0, 1].
    :type plot_ci_alpha: float

    :param plot_ci_make_poly: Function that takes x, y, and their upper and
        lower confidence interval estimations, and returns a plt.Polygon
        correctly for the curve we're trying to draw.
    :type plot_ci_make_poly: (x, x_l, x_u, y, y_l, y_u, **poly_args) ->
        plt.Polygon

    """
    # Create curves for each label and then for overall.

    all_classifications = set()
    for s in label2classifications.values():
        all_classifications.update(s)

    plt.clf()
    plt.figure(figsize=(15, 12), dpi=72)
    for i, l in enumerate(sorted(label2classifications)):
        # record binary truth relation with respect to label `l`
        l_truth = []
        l_proba = []
        # Get classification probability and truth label for all tested elements
        for c in all_classifications:
            l_truth.append(int(c in label2classifications[l]))
            l_proba.append(c[l])
        assert len(l_truth) == len(l_proba) == len(all_classifications), \
            "Somehow didn't wind up with truth/proba values for all " \
            "classification elements"
        x, y, t = skl_curve_func(numpy.array(l_truth), numpy.array(l_proba))
        auc = sklearn.metrics.auc(x, y)
        m = select_color_marker(i)
        plt.plot(x, y, m, label="%s (auc=%f)" % (l, auc))

        if plot_ci:
            # Confidence interval calculation using Wilson's score interval
            x_u, x_l = curve_wilson_ci(x, len(l_proba))
            y_u, y_l = curve_wilson_ci(y, len(l_proba))

            # For each xy point, combine CI upper/lower bounds based on curve
            # being drawn. I.e. zip(x_l, y_l) for PR curve, but zip(x_u, y_l)
            # for ROC curve.
            # TODO: Generate concave hull (alpha shape)
            #       This would be curve direction independent.
            ci_poly = plot_ci_make_poly(x, x_l, x_u, y, y_l, y_u,
                                        facecolor=m[0], edgecolor=m[0],
                                        alpha=plot_ci_alpha)
            plt.gca().add_patch(ci_poly)

    format_plt(title + ' - Class vs. Rest', xlabel, ylabel)
    log.info("Saving: %s", output_filepath)
    plt.savefig(output_filepath)
    plt.clf()


def make_pr_curves(label2classifications, output_filepath, plot_ci,
                   plot_ci_alpha):
    def skl_pr_curve(truth, proba):
        p, r, t = sklearn.metrics.precision_recall_curve(truth, proba,
                                                         pos_label=1)
        return r, p, t

    def make_ci_poly(x, x_l, x_u, y, y_l, y_u, **poly_kwds):
        x_l = numpy.min([x, x_l], 0)
        x_u = numpy.max([x, x_u], 0)
        y_l = numpy.min([y, y_l], 0)
        y_u = numpy.max([y, y_u], 0)
        # Add points to flush ends with plot border
        poly_points = (
            [(x[0], y_l[0])] + zip(x_l, y_l) + [(x[-1], y_l[-1])] +
            [(x[-1], y_u[-1])] + list(reversed(zip(x_u, y_u))) +
            [(x[0], y_u[0])]
        )
        return plt.Polygon(poly_points, **poly_kwds)

    log = logging.getLogger(__name__)
    make_curve(log, skl_pr_curve, "PR", "Recall", "Precision", output_filepath,
               label2classifications, plot_ci, plot_ci_alpha, make_ci_poly)


def make_roc_curves(label2classifications, output_filepath, plot_ci,
                    plot_ci_alpha):
    def skl_roc_curve(truth, proba):
        fpr, tpr, t = sklearn.metrics.roc_curve(truth, proba, pos_label=1)
        return fpr, tpr, t

    def make_ci_poly(x, x_l, x_u, y, y_l, y_u, **poly_kwds):
        x_l = numpy.min([x, x_l], 0)
        x_u = numpy.max([x, x_u], 0)
        y_l = numpy.min([y, y_l], 0)
        y_u = numpy.max([y, y_u], 0)
        poly_points = (
            [(x[0], y_u[0])] + zip(x_l, y_u) + [(x[-1], y_u[-1])] +
            [(x[-1], y_l[-1])] + list(reversed(zip(x_u, y_l))) +
            [(x[0], y_l[0])]
        )
        return plt.Polygon(poly_points, **poly_kwds)

    log = logging.getLogger(__name__)
    make_curve(log, skl_roc_curve, "ROC", "False Positive Rate",
               "True Positive Rate", output_filepath, label2classifications,
               plot_ci, plot_ci_alpha, make_ci_poly)


def curve_wilson_ci(p, n, confidence=0.95):
    """
    Generate upper and lower bounds confidence interval, using the Wilson score
    interval, for PR and ROC curves.

    :param p: Array of points for an axis.
    :type p: numpy.ndarray[float]

    :param n: number of source predictions that led to the ``p`` array.
    :type n: int

    :param confidence: (0, 1) confidence value for interval generation. Default
        of 0.95, or 95% confidence bounds.
    :type confidence: float

    :return: Upper and lower bounds arrays.
    :rtype: (numpy.ndarray[float], numpy.ndarray[float])

    """
    quantile = 1.-(0.5*(1.-confidence))
    z = scipy.stats.norm.ppf(quantile)  # ~1.96 when confidence==0.95
    n = float(n)
    u = (1/(1+(z*z/n))) * \
        (p + (z*z)/(2*n) + z*numpy.sqrt(p * (1 - p) / n + (z*z)/(4*n*n)))
    L = (1/(1+(z*z/n))) * \
        (p + (z*z)/(2*n) - z*numpy.sqrt(p * (1 - p) / n + (z*z)/(4*n*n)))
    return u, L


if __name__ == '__main__':
    main()
