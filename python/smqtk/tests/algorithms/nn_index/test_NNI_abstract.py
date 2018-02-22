from __future__ import division, print_function
import unittest

import mock
import nose.tools as ntools
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
        return super(DummySI, self).build_index(descriptors)

    def nn(self, d, n=1):
        return super(DummySI, self).nn(d, n)

    def count(self):
        return 0


class TestSimilarityIndexAbstract (unittest.TestCase):

    def test_get_impls(self):
        # Some implementations should be returned
        m = get_nn_index_impls()
        ntools.assert_true(m)
        for cls in six.itervalues(m):
            ntools.assert_true(issubclass(cls, NearestNeighborsIndex))

    def test_count(self):
        index = DummySI()
        ntools.assert_equal(index.count(), 0)
        ntools.assert_equal(index.count(), len(index))

        # Pretend that there were things in there. Len should pass it though
        index.count = mock.Mock()
        index.count.return_value = 5
        ntools.assert_equal(len(index), 5)

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
        ntools.assert_raises(ValueError, index.nn, q)

    def test_query_empty_index(self):
        index = DummySI()
        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        ntools.assert_raises(ValueError, index.nn, q)
