import os
import random
import tempfile
import unittest

import nose.tools as ntools
import numpy

from smqtk.data_rep.descriptor_element_impl.local_elements import \
    DescriptorMemoryElement
from smqtk.similarity_index.flann import FlannSimilarity


__author__ = 'purg'


def make_tempfile(suffix=""):
    fd, fp = tempfile.mkstemp(suffix)
    os.close(fd)
    return fp


# Don't bother running tests of the class is not usable
if FlannSimilarity.is_usable():

    class TestFlannIndex (unittest.TestCase):

        FLANN_INDEX_CACHE = None
        FLANN_PARAMETER_CACHE = None
        FLANN_DESCR_CACHE = None

        RAND_SEED = 42

        @classmethod
        def _make_cache_files(cls):
            cls.FLANN_INDEX_CACHE = make_tempfile('.flann')
            cls.FLANN_PARAMETER_CACHE = make_tempfile('.pickle')
            cls.FLANN_DESCR_CACHE = make_tempfile('.pickle')

        @classmethod
        def _clean_cache_files(cls):
            for fp in [cls.FLANN_DESCR_CACHE,
                       cls.FLANN_PARAMETER_CACHE,
                       cls.FLANN_INDEX_CACHE]:
                if fp and os.path.isfile(fp):
                    os.remove(fp)

        def _make_inst(self, dist_method):
            """
            Make an instance of FlannSimilarity
            """
            self._make_cache_files()
            self._clean_cache_files()
            return FlannSimilarity(self.FLANN_INDEX_CACHE,
                                   self.FLANN_PARAMETER_CACHE,
                                   self.FLANN_DESCR_CACHE,
                                   distance_method=dist_method,
                                   random_seed=self.RAND_SEED)

        def tearDown(self):
            self._clean_cache_files()

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
            r, dists = index.nn(q, dim)
            # All dists should be 1.0, r order doesn't matter
            for d in dists:
                ntools.assert_equal(d, 1.)

        def test_known_descriptors_euclidean_ordered(self):
            index = self._make_inst('euclidean')

            # make vectors to return in a known euclidean distance order
            i = 10
            test_descriptors = []
            for j in xrange(i):
                d = DescriptorMemoryElement('ordered', j)
                d.set_vector(numpy.array([j, j*2], float))
                test_descriptors.append(d)
            random.shuffle(test_descriptors)
            index.build_index(test_descriptors)

            # Since descriptors were build in increasing distance from (0,0),
            # returned descriptors for a query of [0,0] should be in index order.
            q = DescriptorMemoryElement('query', 99)
            q.set_vector(numpy.array([0, 0], float))
            r, dists = index.nn(q, i)
            for j, d, dist in zip(range(i), r, dists):
                ntools.assert_equal(d.uuid(), j)

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

        def test_configuration(self):
            self._make_cache_files()
            self._clean_cache_files()

            # Make configuration based on default
            c = FlannSimilarity.default_config()
            c['index_filepath'] = self.FLANN_INDEX_CACHE
            c['parameters_filepath'] = self.FLANN_PARAMETER_CACHE
            c['descriptor_cache_filepath'] = self.FLANN_DESCR_CACHE
            c['distance_method'] = 'hik'
            c['random_seed'] = 42

            # Build based on configuration
            index = FlannSimilarity.from_config(c)
            ntools.assert_equal(index._index_filepath, self.FLANN_INDEX_CACHE)
            ntools.assert_equal(index._index_param_filepath,
                                self.FLANN_PARAMETER_CACHE)
            ntools.assert_equal(index._descr_cache_filepath,
                                self.FLANN_DESCR_CACHE)

            c2 = index.get_config()
            ntools.assert_equal(c, c2)
