from typing import Sequence

import numpy as np

from smqtk.algorithms import SupervisedClassifier
from smqtk.representation import DescriptorElement
from smqtk.representation.descriptor_element.local_elements import (
    DescriptorMemoryElement,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
    cls_conf_to_config_dict,
)

from . import RankRelevancy


class RankRelevancyWithSupervisedClassifier(RankRelevancy):
    """
    Relevancy ranking that utilizes a usable supervised classifier for
    on-the-fly training and inference.

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
        super().__init__()
        self._classifier_type = type(classifier_inst)
        self._classifier_config = classifier_inst.get_config()

    @classmethod
    def get_default_config(cls):
        c = super().get_default_config()
        c['classifier_inst'] = \
            make_default_config(SupervisedClassifier.get_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        config_dict = dict(config_dict)  # shallow copy to write to input dict
        config_dict['classifier_inst'] = \
            from_config_dict(config_dict.get('classifier_inst', {}),
                             SupervisedClassifier.get_impls())
        return super().from_config(
            config_dict, merge_default=merge_default,
        )

    def get_config(self):
        return {
            'classifier_inst':
                cls_conf_to_config_dict(self._classifier_type,
                                        self._classifier_config),
        }

    def rank(
            self,
            pos: Sequence[np.ndarray],
            neg: Sequence[np.ndarray],
            pool: Sequence[np.ndarray],
    ) -> Sequence[float]:
        if len(pool) == 0:
            return []

        # Train supervised classifier with positive/negative examples.
        label_pos = 'pos'
        label_neg = 'neg'

        i = 0

        def create_de(v: np.ndarray) -> DescriptorElement:
            nonlocal i
            # Hopefully type_str doesn't matter
            de = DescriptorMemoryElement('', i)
            de.set_vector(v)
            i += 1
            return de

        classifier = self._classifier_type.from_config(self._classifier_config)
        classifier.train({
            label_pos: map(create_de, pos),
            label_neg: map(create_de, neg),
        })

        # Report ``label_pos`` class probabilities as rank score.
        scores = classifier.classify_arrays(pool)
        return [c_map.get(label_pos, 0.0) for c_map in scores]
