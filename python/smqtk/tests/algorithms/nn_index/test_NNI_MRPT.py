from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from six.moves import range, zip

import random
import unittest

import nose.tools as ntools
import numpy

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms import get_nn_index_impls
from smqtk.algorithms.nn_index.mrpt import MRPTNearestNeighborsIndex

__author__ = "john.moeller@kitware.com"


# Don't bother running tests of the class is not usable
if MRPTNearestNeighborsIndex.is_usable():

    class TestMRPTIndex (unittest.TestCase):

        RAND_SEED = 42

        def _make_inst(self):
            """
            Make an instance of MRPTNearestNeighborsIndex
            """
            return MRPTNearestNeighborsIndex(required_votes=1,
                                             random_seed=self.RAND_SEED)

        def test_impl_findable(self):
            ntools.assert_in(MRPTNearestNeighborsIndex.__name__,
                             get_nn_index_impls())

        def test_known_descriptors_euclidean_unit(self):
            dim = 5

            ###
            # Unit vectors -- Equal distance
            #
            index = self._make_inst()
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, float)
                v[i] = 1.
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            index.build_index(test_descriptors)
            # query descriptor -- zero vector
            # -> all modeled descriptors should be equally distance (unit corners)
            q = DescriptorMemoryElement('query', 0)
            q.set_vector(numpy.zeros(dim, float))
            r, dists = index.nn(q, n=dim)
            ntools.assert_equal(len(dists), dim)
            # All dists should be 1.0, r order doesn't matter
            for d in dists:
                ntools.assert_equal(d, 1.)

        def test_known_descriptors_nearest(self):
            dim = 5

            ###
            # Unit vectors -- Equal distance
            #
            index = self._make_inst()
            test_descriptors = []
            V = numpy.eye(dim, dtype=numpy.float32)
            for i in range(dim):
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(V[i])
                test_descriptors.append(d)
            index.build_index(test_descriptors)
            # query descriptor -- first point
            q = DescriptorMemoryElement('query', 0)
            q.set_vector(V[0])
            r, dists = index.nn(q)
            ntools.assert_equal(len(dists), 1)
            # Distance should be zero
            ntools.assert_equal(dists[0], 0.)
            ntools.assert_items_equal(r[0].vector(), V[0])

        def test_known_descriptors_euclidean_ordered(self):
            index = self._make_inst()

            # make vectors to return in a known euclidean distance order
            i = 100
            test_descriptors = []
            for j in range(i):
                d = DescriptorMemoryElement('ordered', j)
                d.set_vector(numpy.array([j, j*2], float))
                test_descriptors.append(d)
            random.shuffle(test_descriptors)
            index.build_index(test_descriptors)

            # Since descriptors were build in increasing distance from (0,0),
            # returned descriptors for a query of [0,0] should be in index order.
            q = DescriptorMemoryElement('query', 99)
            q.set_vector(numpy.array([0, 0], float))
            r, dists = index.nn(q, n=i)
            # Because the data is one-dimensional, all of the cells will have
            # the same points (any division will just correspond to a point on
            # the line), and a cell can't have more than half of the points
            ntools.assert_equal(len(dists), i//2)
            for j, d, dist in zip(range(i), r, dists):
                ntools.assert_equal(d.uuid(), j)
                numpy.testing.assert_equal(d.vector(), [j, j*2])

        def test_configuration(self):
            index_filepath = '/index_filepath'
            para_filepath = '/param_fp'
            descr_cache_fp = '/descrcachefp'

            # Make configuration based on default
            c = MRPTNearestNeighborsIndex.get_default_config()
            c['index_filepath'] = index_filepath
            c['parameters_filepath'] = para_filepath
            c['descriptor_cache_filepath'] = descr_cache_fp

            # Build based on configuration
            index = MRPTNearestNeighborsIndex.from_config(c)
            ntools.assert_equal(index._index_filepath, index_filepath)
            ntools.assert_equal(index._index_param_filepath, para_filepath)
            ntools.assert_equal(index._descr_cache_filepath, descr_cache_fp)

            c2 = index.get_config()
            ntools.assert_equal(c, c2)
