import mock
import random
import unittest

import nose.tools as ntools
import numpy

from smqtk.algorithms import get_nn_index_impls
from smqtk.algorithms.nn_index.flann import FlannNearestNeighborsIndex
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


# Don't bother running tests of the class is not usable
if FlannNearestNeighborsIndex.is_usable():

    class TestFlannIndex (unittest.TestCase):

        RAND_SEED = 42

        def _make_inst(self, dist_method):
            """
            Make an instance of FlannNearestNeighborsIndex
            """
            return FlannNearestNeighborsIndex(distance_method=dist_method,
                                              random_seed=self.RAND_SEED)

        def test_impl_findable(self):
            # Already here because the implementation is reporting itself as
            # usable.
            ntools.assert_in(FlannNearestNeighborsIndex.__name__,
                             get_nn_index_impls())

        def test_has_model_data_no_uris(self):
            f = FlannNearestNeighborsIndex()
            ntools.assert_false(f._has_model_data())

        def test_has_model_data_empty_elements(self):
            f = FlannNearestNeighborsIndex('', '', '')
            ntools.assert_false(f._has_model_data())

        def test_load_flann_model_empty_data_elements(self):
            # Construct index with valid, but empty, data URIs instances
            empty_data = 'base64://'
            f = FlannNearestNeighborsIndex(empty_data, empty_data, empty_data)
            # Load method should do nothing but set PID since given data was
            # empty.
            f._load_flann_model()
            ntools.assert_is_none(f._descr_cache)
            ntools.assert_is_none(f._flann)
            ntools.assert_is_none(f._flann_build_params)
            ntools.assert_is_not_none(f._pid)

        @mock.patch("smqtk.algorithms.nn_index.flann"
                    ".FlannNearestNeighborsIndex._load_flann_model")
        def test_has_model_data_valid_uris(self, m_flann_lfm):
            # Mocking flann data loading that occurs in constructor when given
            # non-empty URI targets
            f = FlannNearestNeighborsIndex(
                'base64://bW9kZWxEYXRh',  # 'modelData'
                'base64://cGFyYW1EYXRh',  # 'paramData'
                'base64://ZGVzY3JEYXRh',  # 'descrData'
            )
            ntools.assert_true(f._has_model_data())

        def test_known_descriptors_euclidean_unit(self):
            dim = 5

            ###
            # Unit vectors -- Equal distance
            #
            index = self._make_inst('euclidean')
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, float)
                v[i] = 1.
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            index.build_index(test_descriptors)
            # query descriptor -- zero vector
            # -> all modeled descriptors should be equally distance (unit
            #    corners)
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
            for j in range(i):
                d = DescriptorMemoryElement('ordered', j)
                d.set_vector(numpy.array([j, j*2], float))
                test_descriptors.append(d)
            random.shuffle(test_descriptors)
            index.build_index(test_descriptors)

            # Since descriptors were build in increasing distance from (0,0),
            # returned descriptors for a query of [0,0] should be in index
            # order.
            q = DescriptorMemoryElement('query', 99)
            q.set_vector(numpy.array([0, 0], float))
            r, dists = index.nn(q, i)
            for j, d, dist in zip(range(i), r, dists):
                ntools.assert_equal(d.uuid(), j)
                numpy.testing.assert_equal(d.vector(), [j, j*2])

        def test_known_descriptors_hik_unit(self):
            dim = 5

            ###
            # Unit vectors - Equal distance
            #
            index = self._make_inst('hik')
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, float)
                v[i] = 1.
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            index.build_index(test_descriptors)
            # query with zero vector
            # -> all modeled descriptors have no intersection, dists should be
            #    1.0, or maximum distance by histogram intersection.
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
            index_filepath = '/index_filepath'
            para_filepath = '/param_fp'
            descr_cache_fp = '/descrcachefp'

            # Make configuration based on default
            c = FlannNearestNeighborsIndex.get_default_config()
            c['index_uri'] = index_filepath
            c['parameters_uri'] = para_filepath
            c['descriptor_cache_uri'] = descr_cache_fp
            c['distance_method'] = 'hik'
            c['random_seed'] = 42

            # Build based on configuration
            index = FlannNearestNeighborsIndex.from_config(c)
            ntools.assert_equal(index._index_uri, index_filepath)
            ntools.assert_equal(index._index_param_uri, para_filepath)
            ntools.assert_equal(index._descr_cache_uri, descr_cache_fp)

            c2 = index.get_config()
            ntools.assert_equal(c, c2)

        def test_build_index_no_descriptors(self):
            f = FlannNearestNeighborsIndex()
            ntools.assert_raises(
                ValueError,
                f.build_index, []
            )

        def test_build_index(self):
            # Empty memory data elements for storage
            empty_data = 'base64://'
            f = FlannNearestNeighborsIndex(empty_data, empty_data, empty_data)
            # Internal elements should initialize have zero-length byte values
            ntools.assert_equal(len(f._index_elem.get_bytes()), 0)
            ntools.assert_equal(len(f._index_param_elem.get_bytes()), 0)
            ntools.assert_equal(len(f._descr_cache_elem.get_bytes()), 0)

            # Make unit vectors, one for each feature
            dim = 8
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, float)
                v[i] = 1.
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)

            f.build_index(test_descriptors)

            # Internal elements should not have non-zero byte values.
            ntools.assert_greater(len(f._index_elem.get_bytes()), 0)
            ntools.assert_greater(len(f._index_param_elem.get_bytes()), 0)
            ntools.assert_greater(len(f._descr_cache_elem.get_bytes()), 0)
