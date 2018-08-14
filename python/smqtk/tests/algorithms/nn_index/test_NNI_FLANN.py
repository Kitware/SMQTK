from __future__ import division, print_function
import mock
import random
import unittest

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
            self.assertIn(FlannNearestNeighborsIndex.__name__,
                          get_nn_index_impls())

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
            #: :type: FlannNearestNeighborsIndex
            index = FlannNearestNeighborsIndex.from_config(c)
            self.assertEqual(index._index_uri, index_filepath)
            self.assertEqual(index._index_param_uri, para_filepath)
            self.assertEqual(index._descr_cache_uri, descr_cache_fp)

            c2 = index.get_config()
            self.assertEqual(c, c2)

        def test_has_model_data_no_uris(self):
            f = FlannNearestNeighborsIndex()
            self.assertFalse(f._has_model_data())

        def test_has_model_data_empty_elements(self):
            f = FlannNearestNeighborsIndex('', '', '')
            self.assertFalse(f._has_model_data())

        def test_load_flann_model_empty_data_elements(self):
            # Construct index with valid, but empty, data URIs instances
            empty_data = 'base64://'
            f = FlannNearestNeighborsIndex(empty_data, empty_data, empty_data)
            # Load method should do nothing but set PID since given data was
            # empty.
            f._load_flann_model()
            self.assertIsNone(f._descr_cache)
            self.assertIsNone(f._flann)
            self.assertIsNone(f._flann_build_params)
            self.assertIsNotNone(f._pid)

        @mock.patch("smqtk.algorithms.nn_index.flann"
                    ".FlannNearestNeighborsIndex._load_flann_model")
        def test_has_model_data_valid_uris(self, _m_flann_lfm):
            # Mocking flann data loading that occurs in constructor when given
            # non-empty URI targets
            f = FlannNearestNeighborsIndex(
                'base64://bW9kZWxEYXRh',  # 'modelData'
                'base64://cGFyYW1EYXRh',  # 'paramData'
                'base64://ZGVzY3JEYXRh',  # 'descrData'
            )
            self.assertTrue(f._has_model_data())

        def test_build_index_one(self):
            d = DescriptorMemoryElement('test', 0)
            d.set_vector(numpy.zeros(8, float))
            index = self._make_inst('euclidean')
            index.build_index([d])
            self.assertListEqual(
                index._descr_cache,
                [d]
            )
            self.assertIsNotNone(index._flann)
            self.assertIsInstance(index._flann_build_params, dict)

        def test_build_index_with_cache(self):
            # Empty memory data elements for storage
            empty_data = 'base64://'
            f = FlannNearestNeighborsIndex(empty_data, empty_data, empty_data)
            # Internal elements should initialize have zero-length byte values
            self.assertEqual(len(f._index_elem.get_bytes()), 0)
            self.assertEqual(len(f._index_param_elem.get_bytes()), 0)
            self.assertEqual(len(f._descr_cache_elem.get_bytes()), 0)

            # Make unit vectors, one for each feature dimension.
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
            self.assertGreater(len(f._index_elem.get_bytes()), 0)
            self.assertGreater(len(f._index_param_elem.get_bytes()), 0)
            self.assertGreater(len(f._descr_cache_elem.get_bytes()), 0)

        def test_update_index(self):
            # Build index with one descriptor, then "update" with a second
            # different descriptor checking that the new cache contains both.
            d1 = DescriptorMemoryElement('test', 0)
            d1.set_vector(numpy.zeros(8))
            d2 = DescriptorMemoryElement('test', 1)
            d2.set_vector(numpy.ones(8))

            index = self._make_inst('euclidean')
            index.build_index([d1])
            self.assertEqual(index.count(), 1)
            self.assertSetEqual(set(index._descr_cache), {d1})

            index.update_index([d2])
            self.assertEqual(index.count(), 2)
            self.assertSetEqual(set(index._descr_cache), {d1, d2})

        def test_nn_known_descriptors_euclidean_unit(self):
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
                self.assertEqual(d, 1.)

        def test_nn_known_descriptors_euclidean_ordered(self):
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
                self.assertEqual(d.uuid(), j)
                numpy.testing.assert_equal(d.vector(), [j, j*2])

        def test_nn_known_descriptors_hik_unit(self):
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
                self.assertEqual(d, 1.)

            # query with index element
            q = test_descriptors[3]
            r, dists = index.nn(q, 1)
            self.assertEqual(r[0], q)
            self.assertEqual(dists[0], 0.)

            r, dists = index.nn(q, dim)
            self.assertEqual(r[0], q)
            self.assertEqual(dists[0], 0.)

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
            #: :type: FlannNearestNeighborsIndex
            index = FlannNearestNeighborsIndex.from_config(c)
            self.assertEqual(index._index_uri, index_filepath)
            self.assertEqual(index._index_param_uri, para_filepath)
            self.assertEqual(index._descr_cache_uri, descr_cache_fp)

            c2 = index.get_config()
            self.assertEqual(c, c2)

        def test_build_index_no_descriptors(self):
            f = FlannNearestNeighborsIndex()
            self.assertRaises(
                ValueError,
                f.build_index, []
            )

        def test_build_index(self):
            # Empty memory data elements for storage
            empty_data = 'base64://'
            f = FlannNearestNeighborsIndex(empty_data, empty_data, empty_data)
            # Internal elements should initialize have zero-length byte values
            self.assertEqual(len(f._index_elem.get_bytes()), 0)
            self.assertEqual(len(f._index_param_elem.get_bytes()), 0)
            self.assertEqual(len(f._descr_cache_elem.get_bytes()), 0)

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
            self.assertGreater(len(f._index_elem.get_bytes()), 0)
            self.assertGreater(len(f._index_param_elem.get_bytes()), 0)
            self.assertGreater(len(f._descr_cache_elem.get_bytes()), 0)
