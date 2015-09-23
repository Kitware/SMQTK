import unittest

import mock
import nose.tools as ntools
import numpy

from smqtk.representation.descriptor_element_impl.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms.nn_index import NearestNeighborsIndex

__author__ = 'purg'


class DummySI (NearestNeighborsIndex):
    @classmethod
    def is_usable(cls):
        True

    def get_config(self):
        return {}

    def build_index(self, descriptors):
        return super(DummySI, self).build_index(descriptors)

    def nn(self, d, n=1):
        return super(DummySI, self).nn(d, n)

    def count(self):
        return 0


class TestSimilarityIndexAbstract (unittest.TestCase):

    @classmethod
    def setUp(cls):
        # Reset descriptor memory global cache before each test
        DescriptorMemoryElement.MEMORY_CACHE = {}

    @mock.patch.object(DummySI, 'count')
    def test_normal_conditions(self, mock_dsi_count):
        index = DummySI()
        mock_dsi_count.return_value = 1

        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        index.nn(q)

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
