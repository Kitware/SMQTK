import json
import os
import random
import unittest

import nose.tools as ntools
import numpy

from smqtk.representation.code_index.memory import MemoryCodeIndex
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms.nn_index.lsh.itq import ITQNearestNeighborsIndex
from smqtk.utils.file_utils import make_tempfile

__author__ = 'purg'


class TestIqrSimilarityIndex (unittest.TestCase):

    ITQ_ROTATION_MAT = None
    ITQ_MEAN_VEC = None
    RANDOM_SEED = 42

    @classmethod
    def _clean_cache_files(cls):
        for fp in [cls.ITQ_ROTATION_MAT, cls.ITQ_MEAN_VEC]:
            if fp and os.path.isfile(fp):
                os.remove(fp)

    @classmethod
    def _make_cache_files(cls):
        cls._clean_cache_files()
        cls.ITQ_MEAN_VEC = make_tempfile(suffix='.npy')
        cls.ITQ_ROTATION_MAT = make_tempfile(suffix='.npy')

    def _make_inst(self, dist_method, bits=8):
        self._make_cache_files()
        # don't want the files to actually exist
        self._clean_cache_files()
        # Initialize with a fresh code index instance every time, otherwise the
        # same code index is maintained between constructions
        return ITQNearestNeighborsIndex(self.ITQ_MEAN_VEC, self.ITQ_ROTATION_MAT,
                                  code_index=MemoryCodeIndex(),
                                  bit_length=bits,
                                  distance_method=dist_method,
                                  random_seed=self.RANDOM_SEED)

    def tearDown(self):
        self._clean_cache_files()

    def test_configuration(self):
        c = ITQNearestNeighborsIndex.get_default_config()
        # Default code index should be memory based
        ntools.assert_equal(c['code_index']['type'], 'MemoryCodeIndex')
        ntools.assert_true(c['mean_vec_filepath'] is None)
        ntools.assert_true(c['rotation_filepath'] is None)
        ntools.assert_true(c['random_seed'] is None)

        # Conversion to JSON and back is idempotent
        ntools.assert_equal(json.loads(json.dumps(c)), c)

        # Make some changes to deviate from defaults
        c['bit_length'] = 256
        c['itq_iterations'] = 25
        c['mean_vec_filepath'] = 'vec.npy'
        c['rotation_filepath'] = 'rot.npy'

        # Make instance
        index = ITQNearestNeighborsIndex.from_config(c)
        ntools.assert_equal(index._mean_vec_cache_filepath,
                            c['mean_vec_filepath'])
        ntools.assert_equal(index._rotation_cache_filepath,
                            c['rotation_filepath'])
        ntools.assert_is_instance(index._code_index, MemoryCodeIndex)
        ntools.assert_equal(index._bit_len, c['bit_length'])
        ntools.assert_equal(index._itq_iter_num, c['itq_iterations'])
        ntools.assert_equal(index._dist_method, c['distance_method'])
        ntools.assert_equal(index._rand_seed, c['random_seed'])

    def test_known_descriptors_euclidean_unit(self):
        dim = 5

        ###
        # Unit vectors -- Equal distance
        #
        index = self._make_inst('euclidean')
        test_descriptors = []
        for i in xrange(dim):
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
        # All dists should be 1.0, r order doesn't matter
        r, dists = index.nn(q, dim)
        for d in dists:
            ntools.assert_equal(d, 1.)

    def test_known_descriptors_euclidean_ordered(self):
        index = self._make_inst('euclidean')

        # make vectors to return in a known euclidean distance order
        i = 1000
        test_descriptors = []
        for j in xrange(i):
            d = DescriptorMemoryElement('ordered', j)
            d.set_vector(numpy.array([j, j*2], float))
            test_descriptors.append(d)
        random.shuffle(test_descriptors)
        index.build_index(test_descriptors)

        # Since descriptors were build in increasing distance from (0,0),
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

    def test_random_descriptors_euclidean(self):
        # make random descriptors
        i = 1000
        dim = 256
        bits = 32
        td = []
        for j in xrange(i):
            d = DescriptorMemoryElement('random', j)
            d.set_vector(numpy.random.rand(dim))
            td.append(d)

        index = self._make_inst('euclidean', bits)
        index.build_index(td)

        # test query from build set -- should return same descriptor when k=1
        q = td[255]
        r, dists = index.nn(q, 1)
        ntools.assert_equal(r[0], q)

        # test query very near a build vector
        td_q = td[0]
        q = DescriptorMemoryElement('query', i)
        v = numpy.array(td_q.vector())  # copy
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
        for j in xrange(1, len(dists)):
            ntools.assert_greater(dists[j], dists[j-1])
        r, dists = index.nn(q, i)
        for j in xrange(1, len(dists)):
            ntools.assert_greater(dists[j], dists[j-1])

    def test_known_descriptors_hik_unit(self):
        dim = 5

        ###
        # Unit vectors - Equal distance
        #
        index = self._make_inst('hik')
        test_descriptors = []
        for i in xrange(dim):
            v = numpy.zeros(dim, float)
            v[i] = 1.
            d = DescriptorMemoryElement('unit', i)
            d.set_vector(v)
            test_descriptors.append(d)
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
