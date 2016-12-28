import os
import unittest

import nose.tools

from smqtk.algorithms.classifier import get_classifier_impls
from smqtk.algorithms.classifier.index_label import IndexLabelClassifier
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.tests import TEST_DATA_DIR


class TestIndexLabelClassifier (unittest.TestCase):

    EXPECTED_LABEL_VEC = [
        'label_1',
        'label_2',
        'negative',
        'label_3',
        'Kitware',
        'label_4',
    ]

    FILEPATH_TEST_LABELS = os.path.join(TEST_DATA_DIR, 'test_labels.txt')

    def test_is_usable(self):
        # Should always be available
        nose.tools.assert_true(IndexLabelClassifier.is_usable())

    def test_impl_findable(self):
        nose.tools.assert_in(IndexLabelClassifier.__name__,
                             get_classifier_impls())

    def test_new(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        nose.tools.assert_equal(c.label_vector, self.EXPECTED_LABEL_VEC)

    def test_get_labels(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        nose.tools.assert_equal(c.get_labels(), self.EXPECTED_LABEL_VEC)

    def test_configuration(self):
        cfg = IndexLabelClassifier.get_default_config()
        nose.tools.assert_equal(cfg, {'index_to_label_uri': None})

        cfg['index_to_label_uri'] = self.FILEPATH_TEST_LABELS
        c = IndexLabelClassifier.from_config(cfg)
        nose.tools.assert_equal(c.get_config(), cfg)

    def test_classify(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        m_expected = {
            'label_1': 1,
            'label_2': 2,
            'negative': 3,
            'label_3': 4,
            'Kitware': 5,
            'label_4': 6,
        }

        d = DescriptorMemoryElement('test', 0)
        d.set_vector([1, 2, 3, 4, 5, 6])

        m = c._classify(d)
        nose.tools.assert_equal(m, m_expected)

    def test_classify_invalid_descriptor_dimensions(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        d = DescriptorMemoryElement('test', 0)

        # One less
        d.set_vector([1, 2, 3, 4, 5])
        nose.tools.assert_raises(
            RuntimeError,
            c._classify, d
        )

        # One more
        d.set_vector([1, 2, 3, 4, 5, 6, 7])
        nose.tools.assert_raises(
            RuntimeError,
            c._classify, d
        )
