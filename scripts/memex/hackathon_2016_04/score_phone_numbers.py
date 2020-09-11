#!/usr/bin/env python

import csv
import json
import logging

from matplotlib import pyplot as plt
import numpy
import six
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from smqtk.algorithms import get_classifier_impls
from smqtk.representation import ClassificationElementFactory
from smqtk.representation.classification_element.memory import MemoryClassificationElement
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.descriptor_set.memory import MemoryDescriptorSet
from smqtk.utils.cli import initialize_logging
from smqtk.utils.plugin import from_plugin_config


initialize_logging(logging.getLogger(), logging.INFO)
log = logging.getLogger(__name__)


###############################################################################
# Parameters
#
PHONE_SHA1_JSON = "eval.map.phone2shas.json"
DESCRIPTOR_SET_FILE_CACHE = "eval.images.descriptors.alexnet_fc7.index"

CLASSIFIER_TRAINING_CONFIG_JSON = 'ad-images.final.cmv.train.json'

PHONE2SCORE_OUTPUT_FILEPATH = "eval.results.full_model.phone2score.csv"

# Optional for ROC generation, using PHONE2SCORE_OUTPUT_FILEPATH as input, and
# outputting plots
PHONE2TRUTH = 'eval.source.phone2truth.json'
PLOT_CM_OUTPUT = 'eval.results.full_model.plot.cm.png'
PLOT_ROC_OUTPUT = 'eval.results.full_model.plot.roc.png'
PLOT_PR_OUTPUT = 'eval.results.full_model.plot.pr.png'


###############################################################################
# Code
#

if isinstance(PHONE2TRUTH, basestring) and PHONE2TRUTH:
    phone2score = dict((p, float(s))
                       for p, s
                       in csv.reader(open(PHONE2SCORE_OUTPUT_FILEPATH)))
    phone2truth = json.load(open(PHONE2TRUTH))
    ordered_phones = sorted(phone2score)
    T = 0.5

    v_truth, v_proba = zip(*[(phone2truth[p], phone2score[p])
                             for p in ordered_phones])
    v_predicted = [((s >= T and 'positive') or 'negative') for s in v_proba]

    # Confusion Matrix
    log.info("Constructing confusion matrix")
    labels = ['positive', 'negative']
    cm = confusion_matrix(v_truth, v_predicted, labels)
    cm_norm = cm / cm.sum(1).astype(float)[:, numpy.newaxis]
    plt.clf()
    f = plt.figure(0)
    ax = f.add_subplot(111)
    cax = ax.matshow(cm_norm)
    f.colorbar(cax)
    # Annotate cells with count values
    for y in xrange(cm.shape[0]):
        for x in xrange(cm.shape[1]):
            ax.annotate(s=str(cm[y, x]), xy=(x, y), xycoords='data')
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set_title('Confusion Matrix - Count Makeup')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    f.savefig(PLOT_CM_OUTPUT)


    log.info("Constructing PR/ROC curve")
    # x, y, ignore
    fpr, tpr, _ = roc_curve(v_truth, v_proba, pos_label='positive')
    roc_curve_area = auc(fpr, tpr)
    # y, x, ignore
    p, r, _ = precision_recall_curve(v_truth, v_proba, pos_label='positive')
    pr_curve_area = auc(r, p)
    # ROC Curve
    plt.clf()
    plt.plot(fpr, tpr, label="auc=%f" % roc_curve_area)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True positive rate")
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.savefig(PLOT_ROC_OUTPUT)
    # PR Curve
    plt.clf()
    plt.plot(r, p, label="auc=%f" % pr_curve_area)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.title("PR - HT Positive")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.savefig(PLOT_PR_OUTPUT)

else:
    # Using the final trained classifier
    with open(CLASSIFIER_TRAINING_CONFIG_JSON) as f:
        classifier_config = json.load(f)

    log.info("Loading plugins")
    descr_set_cache_elem = DataFileElement(filepath=DESCRIPTOR_SET_FILE_CACHE)
    descriptor_set = MemoryDescriptorSet(cache_element=descr_set_cache_elem)
    #: :type: smqtk.algorithms.Classifier
    classifier = from_plugin_config(classifier_config['plugins']['classifier'],
                                    get_classifier_impls())
    c_factory = ClassificationElementFactory(MemoryClassificationElement, {})

    #: :type: dict[str, list[str]]
    phone2shas = json.load(open(PHONE_SHA1_JSON))
    #: :type: dict[str, float]
    phone2score = {}

    log.info("Classifying phone imagery descriptors")
    i = 0
    descriptor_set_shas = set(descriptor_set.iterkeys())
    for p in phone2shas:
        log.info('%s (%d / %d)', p, i + 1, len(phone2shas))
        # Not all source "images" have descriptors since some URLs returned
        # non-image files. Intersect phone sha's with what was actually
        # computed. Warn if this reduces descriptors for classification to zero.
        indexed_shas = set(phone2shas[p]) & descriptor_set_shas
        if not indexed_shas:
            raise RuntimeError(
                "Phone number '%s' has no valid images associated "
                "with it.\nBefore:\n%s\n\nAfter:\n%s"
                % (p, phone2shas[p], indexed_shas))

        descriptor_elems = descriptor_set.get_many_descriptors(*indexed_shas)
        pos_scores = [c['positive'] for c
                      in classifier.classify_elements(
                          descriptor_elems, c_factory,
                      )]

        # Max of pool
        phone2score[p] = max(pos_scores)

        i += 1

    log.info("Saving score map")
    csv.writer(open(PHONE2SCORE_OUTPUT_FILEPATH, 'w')) \
        .writerows(sorted(six.iteritems(phone2score)))


log.info("Done")
