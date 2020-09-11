import numpy as np
from six.moves import zip

from smqtk.algorithms import SupervisedClassifier
from smqtk.representation import DescriptorElement
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
    cls_conf_to_config_dict,
)

from . import RelevancyIndex, NoIndexError


class SupervisedClassifierRelevancyIndex (RelevancyIndex):
    """
    Relevancy index that utilizes a usable supervised classifier for on-the-fly
    training and inference.

    # Classifier "cloning"
    The input supervised classifier instance to the constructor is not directly
    used, but its type and configuration are recorded in order to create a new
    instance in ``rank`` to train and classify the index.

    The caveat here is that any non-configuration reflected, runtime
    modifications to the input classifier will not be reflected by the
    classifier used in ``rank``.

    Using a copy of the input classifier allows the ``rank`` method to be used
    in parallel without blocking other calls to ``rank``.

    :param smqtk.algorithms.SupervisedClassifier classifier_inst:
        Supervised classifier instance to base the ephemeral ranking classifier
        on. The type and configuration of this classifier is used to create a
        clone at rank time. The input classifier instance is not modified.
    """

    def __init__(self, classifier_inst):
        super(SupervisedClassifierRelevancyIndex, self).__init__()
        self._classifier_type = type(classifier_inst)
        self._classifier_config = classifier_inst.get_config()
        # Some number of descriptors to be ranked, cached as the elements
        # themselves as well as a vertically-stacked matrix (ndim==2).
        # These are None when there is no index yet.
        self._descr_elem_list = None
        self._descr_matrix = None

    @classmethod
    def is_usable(cls):
        # This being a wrapper to other plugins, this is always available
        # and is more contingent on nested implementations existing.
        return True

    @classmethod
    def get_default_config(cls):
        c = super(SupervisedClassifierRelevancyIndex, cls).get_default_config()
        c['classifier_inst'] = \
            make_default_config(SupervisedClassifier.get_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        config_dict = dict(config_dict)  # shallow copy to write to input dict
        config_dict['classifier_inst'] = \
            from_config_dict(config_dict.get('classifier_inst', {}),
                             SupervisedClassifier.get_impls())
        return super(SupervisedClassifierRelevancyIndex, cls).from_config(
            config_dict, merge_default=merge_default
        )

    def get_config(self):
        return {
            'classifier_inst':
                cls_conf_to_config_dict(self._classifier_type,
                                        self._classifier_config),
        }

    def count(self):
        return 0 if self._descr_matrix is None else len(self._descr_matrix)

    def build_index(self, descriptors):
        # Cache given descriptor element vectors into a matrix for use during
        # ``rank``.
        descr_elem_list = list(descriptors)
        if len(descr_elem_list) == 0:
            raise ValueError("No descriptor elements passed.")
        # note: this fails if multiple descriptor elements with the same UID
        #       are included. There will be None's present.
        descr_matrix = np.asarray(
            DescriptorElement.get_many_vectors(descr_elem_list)
        )
        # If the result matrix is of dtype(object), then either some elements
        # did not have vectors or some vectors were not of congruent
        # dimensionality.
        if descr_matrix.dtype == np.dtype(object):
            raise ValueError("One or more descriptor elements did not have a "
                             "vector set or were of congruent dimensionality.")
        self._descr_elem_list = descr_elem_list
        self._descr_matrix = descr_matrix

    def rank(self, pos, neg):
        if self._descr_elem_list is None or self._descr_matrix is None:
            raise NoIndexError("No index built before calling rank.")

        # Train supervised classifier with positive/negative examples.
        label_pos = 'pos'
        label_neg = 'neg'

        classifier = self._classifier_type.from_config(self._classifier_config)
        classifier.train({
            label_pos: pos,
            label_neg: neg,
        })

        # Report ``label_pos`` class probabilities as rank score.
        cd_iter = zip(classifier.classify_arrays(self._descr_matrix),
                      self._descr_elem_list)
        report = {}
        for c_map, d_elem in cd_iter:
            report[d_elem] = c_map.get(label_pos, 0.0)
        return report
