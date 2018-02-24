from __future__ import division, print_function
import json
import random
import types
import unittest

import nose.tools as ntools
import numpy

from smqtk.algorithms.nn_index.lsh import LSHNearestNeighborIndex
from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.algorithms.nn_index.hash_index.linear import LinearHashIndex
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex
from smqtk.representation.key_value.memory import MemoryKeyValueStore


class TestLshIndex (unittest.TestCase):

    def test_is_usable(self):
        # Should always be usable since this is a shell class.
        ntools.assert_true(LSHNearestNeighborIndex.is_usable())

    def test_configuration(self):
        c = LSHNearestNeighborIndex.get_default_config()

        # Check that default is in JSON format and is decoded to the same
        # result.
        ntools.assert_equal(json.loads(json.dumps(c)), c)

        # Make a simple configuration
        c['lsh_functor']['type'] = 'ItqFunctor'
        c['descriptor_index']['type'] = 'MemoryDescriptorIndex'
        c['hash2uuids_kvstore']['type'] = 'MemoryKeyValueStore'
        c['hash_index']['type'] = 'LinearHashIndex'
        index = LSHNearestNeighborIndex.from_config(c)

        ntools.assert_is_instance(index.lsh_functor,
                                  ItqFunctor)
        ntools.assert_is_instance(index.descriptor_index,
                                  MemoryDescriptorIndex)
        ntools.assert_is_instance(index.hash_index,
                                  LinearHashIndex)
        ntools.assert_is_instance(index.hash2uuids_kvstore,
                                  MemoryKeyValueStore)

        # Can convert instance config to JSON
        ntools.assert_equal(
            json.loads(json.dumps(index.get_config())),
            index.get_config()
        )

    def test_configuration_none_HI(self):
        c = LSHNearestNeighborIndex.get_default_config()

        # Check that default is in JSON format and is decoded to the same
        # result.
        ntools.assert_equal(json.loads(json.dumps(c)), c)

        # Make a simple configuration
        c['lsh_functor']['type'] = 'ItqFunctor'
        c['descriptor_index']['type'] = 'MemoryDescriptorIndex'
        c['hash2uuids_kvstore']['type'] = 'MemoryKeyValueStore'
        c['hash_index']['type'] = None
        index = LSHNearestNeighborIndex.from_config(c)

        ntools.assert_is_instance(index.lsh_functor,
                                  ItqFunctor)
        ntools.assert_is_instance(index.descriptor_index,
                                  MemoryDescriptorIndex)
        ntools.assert_is_none(index.hash_index)
        ntools.assert_is_instance(index.hash2uuids_kvstore,
                                  MemoryKeyValueStore)

        # Can convert instance config to JSON
        ntools.assert_equal(
            json.loads(json.dumps(index.get_config())),
            index.get_config()
        )

    def test_get_dist_func_euclidean(self):
        f = LSHNearestNeighborIndex._get_dist_func('euclidean')
        ntools.assert_is_instance(f, types.FunctionType)
        ntools.assert_almost_equal(
            f(numpy.array([0, 0]), numpy.array([0, 1])),
            1.0
        )

    def test_get_dist_func_cosine(self):
        f = LSHNearestNeighborIndex._get_dist_func('cosine')
        ntools.assert_is_instance(f, types.FunctionType)
        ntools.assert_almost_equal(
            f(numpy.array([1, 0]), numpy.array([0, 1])),
            1.0
        )
        ntools.assert_almost_equal(
            f(numpy.array([1, 0]), numpy.array([1, 1])),
            0.5
        )

    def test_get_dist_func_hik(self):
        f = LSHNearestNeighborIndex._get_dist_func('hik')
        ntools.assert_is_instance(f, types.FunctionType)
        ntools.assert_almost_equal(
            f(numpy.array([0, 0]), numpy.array([0, 1])),
            1.0
        )
        ntools.assert_almost_equal(
            f(numpy.array([1, 0]), numpy.array([0, 1])),
            1.0
        )
        ntools.assert_almost_equal(
            f(numpy.array([1, 1]), numpy.array([0, 1])),
            0.0
        )

    def test_build_index_read_only(self):
        index = LSHNearestNeighborIndex(ItqFunctor(), MemoryDescriptorIndex(),
                                        MemoryKeyValueStore(), read_only=True)
        ntools.assert_raises(
            ReadOnlyError,
            index.build_index, []
        )


class TestLshIndexAlgorithms (unittest.TestCase):

    RANDOM_SEED = 0

    def _make_ftor_itq(self, bits=32):
        itq_ftor = ItqFunctor(bit_length=bits, random_seed=self.RANDOM_SEED)

        def itq_fit(D):
            itq_ftor.fit(D)

        return itq_ftor, itq_fit

    def _make_hi_linear(self):
        return LinearHashIndex()

    def _make_hi_balltree(self):
        return SkLearnBallTreeHashIndex(random_seed=self.RANDOM_SEED)

    #
    # Test LSH with random vectors
    #
    def _random_euclidean(self, hash_ftor, hash_idx,
                          ftor_train_hook=lambda d: None):
        # :param hash_ftor: Hash function class for generating hash codes for
        #   descriptors.
        # :param hash_idx: Hash index instance to use in local LSH algo
        #   instance.
        # :param ftor_train_hook: Function for training functor if necessary.

        # make random descriptors
        i = 1000
        dim = 256
        td = []
        numpy.random.seed(self.RANDOM_SEED)
        for j in range(i):
            d = DescriptorMemoryElement('random', j)
            d.set_vector(numpy.random.rand(dim))
            td.append(d)

        ftor_train_hook(td)

        di = MemoryDescriptorIndex()
        kvstore = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(hash_ftor, di, kvstore,
                                        hash_index=hash_idx,
                                        distance_method='euclidean')
        index.build_index(td)

        # test query from build set -- should return same descriptor when k=1
        q = td[255]
        r, dists = index.nn(q, 1)
        ntools.assert_equal(r[0], q)

        # test query very near a build vector
        td_q = td[0]
        q = DescriptorMemoryElement('query', i)
        v = td_q.vector().copy()
        v_min = max(v.min(), 0.1)
        v[0] += v_min
        v[dim-1] -= v_min
        q.set_vector(v)
        r, dists = index.nn(q, 1)
        ntools.assert_false(numpy.array_equal(q.vector(), td_q.vector()))
        ntools.assert_equal(r[0], td_q)

        # random query
        q = DescriptorMemoryElement('query', i+1)
        q.set_vector(numpy.random.rand(dim))

        # for any query of size k, results should at least be in distance order
        r, dists = index.nn(q, 10)
        for j in range(1, len(dists)):
            ntools.assert_greater(dists[j], dists[j-1])
        r, dists = index.nn(q, i)
        for j in range(1, len(dists)):
            ntools.assert_greater(dists[j], dists[j-1])

    def test_random_euclidean__itq__None(self):
        ftor, fit = self._make_ftor_itq()
        self._random_euclidean(ftor, None, fit)

    def test_random_euclidean__itq__linear(self):
        ftor, fit = self._make_ftor_itq()
        hi = self._make_hi_linear()
        self._random_euclidean(ftor, hi, fit)

    def test_random_euclidean__itq__balltree(self):
        ftor, fit = self._make_ftor_itq()
        hi = self._make_hi_balltree()
        self._random_euclidean(ftor, hi, fit)

    #
    # Test unit vectors
    #
    def _known_unit(self, hash_ftor, hash_idx, dist_method,
                    ftor_train_hook=lambda d: None):
        ###
        # Unit vectors - Equal distance
        #
        dim = 5
        test_descriptors = []
        for i in range(dim):
            v = numpy.zeros(dim, float)
            v[i] = 1.
            d = DescriptorMemoryElement('unit', i)
            d.set_vector(v)
            test_descriptors.append(d)

        ftor_train_hook(test_descriptors)

        di = MemoryDescriptorIndex()
        kvstore = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(hash_ftor, di, kvstore,
                                        hash_index=hash_idx,
                                        distance_method=dist_method)
        index.build_index(test_descriptors)

        # query with zero vector
        # -> all modeled descriptors have no intersection, dists should be 1.0,
        #    or maximum distance by histogram intersection
        q = DescriptorMemoryElement('query', 0)
        q.set_vector(numpy.zeros(dim, float))
        r, dists = index.nn(q, dim)
        # All dists should be 1.0, r order doesn't matter
        for d in dists:
            ntools.assert_equal(d, 1.)

        # query with index element
        q = test_descriptors[3]
        r, dists = index.nn(q, 1)
        ntools.assert_equal(r[0], q)
        ntools.assert_equal(dists[0], 0.)

        r, dists = index.nn(q, dim)
        ntools.assert_equal(r[0], q)
        ntools.assert_equal(dists[0], 0.)

    def test_known_unit__euclidean__itq__None(self):
        ftor, fit = self._make_ftor_itq(5)
        self._known_unit(ftor, None, 'euclidean', fit)

    def test_known_unit__hik__itq__None(self):
        ftor, fit = self._make_ftor_itq(5)
        self._known_unit(ftor, None, 'hik', fit)

    def test_known_unit__euclidean__itq__linear(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_linear()
        self._known_unit(ftor, hi, 'euclidean', fit)

    def test_known_unit__hik__itq__linear(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_linear()
        self._known_unit(ftor, hi, 'hik', fit)

    def test_known_unit__euclidean__itq__balltree(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_balltree()
        self._known_unit(ftor, hi, 'euclidean', fit)

    def test_known_unit__hik__itq__balltree(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_balltree()
        self._known_unit(ftor, hi, 'hik', fit)

    #
    # Test with known vectors and euclidean dist
    #
    def _known_ordered_euclidean(self, hash_ftor, hash_idx,
                                 ftor_train_hook=lambda d: None):
        # make vectors to return in a known euclidean distance order
        i = 1000
        test_descriptors = []
        for j in range(i):
            d = DescriptorMemoryElement('ordered', j)
            d.set_vector(numpy.array([j, j*2], float))
            test_descriptors.append(d)
        random.shuffle(test_descriptors)

        ftor_train_hook(test_descriptors)

        di = MemoryDescriptorIndex()
        kvstore = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(hash_ftor, di, kvstore,
                                        hash_index=hash_idx,
                                        distance_method='euclidean')
        index.build_index(test_descriptors)

        # Since descriptors were built in increasing distance from (0,0),
        # returned descriptors for a query of [0,0] should be in index order.
        q = DescriptorMemoryElement('query', i)
        q.set_vector(numpy.array([0, 0], float))
        # top result should have UUID == 0 (nearest to query)
        r, dists = index.nn(q, 5)
        ntools.assert_equal(r[0].uuid(), 0)
        ntools.assert_equal(r[1].uuid(), 1)
        ntools.assert_equal(r[2].uuid(), 2)
        ntools.assert_equal(r[3].uuid(), 3)
        ntools.assert_equal(r[4].uuid(), 4)
        # global search should be in complete order
        r, dists = index.nn(q, i)
        for j, d, dist in zip(range(i), r, dists):
            ntools.assert_equal(d.uuid(), j)

    def test_known_ordered_euclidean__itq__None(self):
        ftor, fit = self._make_ftor_itq(1)
        self._known_ordered_euclidean(ftor, None, fit)

    def test_known_ordered_euclidean__itq__linear(self):
        ftor, fit = self._make_ftor_itq(1)
        hi = self._make_hi_linear()
        self._known_ordered_euclidean(ftor, hi, fit)

    def test_known_ordered_euclidean__itq__balltree(self):
        ftor, fit = self._make_ftor_itq(1)
        hi = self._make_hi_balltree()
        self._known_ordered_euclidean(ftor, hi, fit)
