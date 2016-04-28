#!/usr/bin/env python

import csv
import json
import logging

from smqtk.algorithms import get_classifier_impls
from smqtk.representation import ClassificationElementFactory
from smqtk.representation.classification_element.memory import MemoryClassificationElement
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex
from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.plugin import from_plugin_config


initialize_logging(logging.getLogger(), logging.INFO)
log = logging.getLogger(__name__)


###############################################################################
# Parameters
#
PHONE_SHA1_JSON = "ad-images.fold_test.map.phone2shas.json"
DESCRIPTOR_INDEX_FILE_CACHE = "ad-images.descriptors.alexnet_fc7.index"

PHONE2SCORE_OUTPUT_FILEPATH = "ad-images.fold_test.map.phone2pos_score.csv"

# Optional for ROC generation
PHONE2TRUTH = None


###############################################################################
# Code
#

# Using the final trained classifier
with open('ad-images.final.cmv.train.json') as f:
    TRAIN_CONFIG = json.load(f)

log.info("Loading plugins")
descriptor_index = MemoryDescriptorIndex(file_cache=DESCRIPTOR_INDEX_FILE_CACHE)
#: :type: smqtk.algorithms.Classifier
classifier = from_plugin_config(TRAIN_CONFIG['plugins']['classifier'],
                                get_classifier_impls())
c_factory = ClassificationElementFactory(MemoryClassificationElement, {})

#: :type: dict[str, list[str]]
phone2shas = json.load(open(PHONE_SHA1_JSON))
#: :type: dict[str, float]
phone2score = {}

log.info("Classifying phone imagery descriptors")
i = 0
for p in phone2shas:
    log.info('%s (%d / %d)', p, i+1, len(phone2shas))
    descriptor_elems = descriptor_index.get_many_descriptors(*phone2shas[p])
    e2c = classifier.classify_async(descriptor_elems, c_factory,
                                    use_multiprocessing=True, ri=1.)
    pos_scores = [c['positive'] for c in e2c.values()]

    # Max of pool
    phone2score[p] = max(pos_scores)

    i += 1

log.info("Saving score map")
csv.writer(open(PHONE2SCORE_OUTPUT_FILEPATH, 'w')) \
   .writerows(sorted(phone2score.iteritems()))

if isinstance(PHONE2TRUTH, basestring):
    from matplotlib import pyplot as plt
    from sklearn.metrics import auc, confusion_matrix, roc_curve

    phone2truth = json.load(open(PHONE2TRUTH))
    ordered_phones = sorted(phone2score)
    T = 0.618

    v_truth, v_proba = zip(*[(phone2truth[p], phone2score[p])
                             for p in ordered_phones])
    v_predicted = [((s >= T and 'positive') or 'negative')
                   for s in v_proba]

    # Confusion Matrix
    labels = ['positive', 'negative']
    cm = confusion_matrix(v_truth, v_predicted, labels)

    # ROC Curve
    # x, y, ignore
    fpr, tpr, t = roc_curve(v_truth, v_proba, pos_label='positive')
    curve_area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="auc=%f" % curve_area)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True positive rate")
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    # plt.show()


log.info("Done")
