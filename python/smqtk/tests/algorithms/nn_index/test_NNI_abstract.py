import unittest

import mock
import numpy
import six

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms.nn_index import NearestNeighborsIndex, get_nn_index_impls


class DummySI (NearestNeighborsIndex):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def build_index(self, descriptors):
        super(DummySI, self).build_index(descriptors)

    def update_index(self, descriptors):
        super(DummySI, self).update_index(descriptors)

    def nn(self, d, n=1):
        return super(DummySI, self).nn(d, n)

    def count(self):
        return 0


class TestSimilarityIndexAbstract (unittest.TestCase):

    def test_get_impls(self):
        # Some implementations should be returned
        m = get_nn_index_impls()
        self.assertTrue(m)
        for cls in six.itervalues(m):
            self.assertTrue(issubclass(cls, NearestNeighborsIndex))

    def test_count(self):
        index = DummySI()
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.count(), len(index))

        # Pretend that there were things in there. Len should pass it though
        index.count = mock.Mock()
        index.count.return_value = 5
        self.assertEqual(len(index), 5)

    def test_build_index_no_descriptors(self):
        index = DummySI()
        self.assertRaises(
            ValueError,
            index.build_index,
            []
        )

    def test_build_index_nonzero_descriptors(self):
        index = DummySI()
        d = DescriptorMemoryElement('test', 0)
        index.build_index([d])

    def test_update_index_no_descriptors(self):
        index = DummySI()
        self.assertRaises(
            ValueError,
            index.update_index,
            []
        )

    def test_update_index_nonzero_descriptors(self):
        index = DummySI()
        d = DescriptorMemoryElement('test', 0)
        index.update_index([d])

    # noinspection PyUnresolvedReferences
    @mock.patch.object(DummySI, 'count')
    def test_normal_conditions(self, mock_dsi_count):
        index = DummySI()
        mock_dsi_count.return_value = 1

        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        index.nn(q)

    # noinspection PyUnresolvedReferences
    @mock.patch.object(DummySI, 'count')
    def test_query_empty_value(self, mock_dsi_count):
        # distance method doesn't matter
        index = DummySI()
        # pretend that we have an index of some non-zero size
        mock_dsi_count.return_value = 1

        # intentionally empty
        q = DescriptorMemoryElement('q', 0)
        self.assertRaises(ValueError, index.nn, q)

    def test_query_empty_index(self):
        index = DummySI()
        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        self.assertRaises(ValueError, index.nn, q)
