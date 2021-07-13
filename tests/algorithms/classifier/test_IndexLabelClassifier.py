from __future__ import division, print_function
import os
import unittest

import numpy
import six
from six.moves import map
import pytest

from smqtk.algorithms.classifier import Classifier
from smqtk.algorithms.classifier.index_label import IndexLabelClassifier
from smqtk.utils.configuration import configuration_test_helper

from tests import TEST_DATA_DIR


class TestIndexLabelClassifier (unittest.TestCase):

    EXPECTED_LABEL_VEC = list(map(six.b, [
        'label_1',
        'label_2',
        'negative',
        'label_3',
        'Kitware',
        'label_4',
    ]))

    FILEPATH_TEST_LABELS = os.path.join(TEST_DATA_DIR, 'test_labels.txt')

    def test_is_usable(self):
        # Should always be available
        self.assertTrue(IndexLabelClassifier.is_usable())

    def test_impl_findable(self):
        self.assertIn(IndexLabelClassifier,
                      Classifier.get_impls())

    def test_configurable(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        for inst in configuration_test_helper(c):
            assert inst.index_to_label_uri == self.FILEPATH_TEST_LABELS

    def test_new(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        self.assertEqual(c.label_vector, self.EXPECTED_LABEL_VEC)

    def test_get_labels(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        self.assertEqual(c.get_labels(), self.EXPECTED_LABEL_VEC)

    def test_configuration(self):
        cfg = IndexLabelClassifier.get_default_config()
        self.assertEqual(cfg, {'index_to_label_uri': None})

        cfg['index_to_label_uri'] = self.FILEPATH_TEST_LABELS
        c = IndexLabelClassifier.from_config(cfg)
        self.assertEqual(c.get_config(), cfg)

    def test_classify_arrays(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)
        c_expected = {
            six.b('label_1'): 1,
            six.b('label_2'): 2,
            six.b('negative'): 3,
            six.b('label_3'): 4,
            six.b('Kitware'): 5,
            six.b('label_4'): 6,
        }

        a = numpy.array([1, 2, 3, 4, 5, 6])
        c = list(c._classify_arrays([a]))[0]
        self.assertEqual(c, c_expected)

    def test_classify_arrays_invalid_descriptor_dimensions(self):
        c = IndexLabelClassifier(self.FILEPATH_TEST_LABELS)

        # One less
        a = numpy.array([1, 2, 3, 4, 5])
        with pytest.raises(RuntimeError):
            list(c._classify_arrays([a]))

        # One more
        a = numpy.array([1, 2, 3, 4, 5, 6, 7])
        with pytest.raises(RuntimeError):
            list(c._classify_arrays([a]))
