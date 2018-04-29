from __future__ import division, print_function
import mock
import random
import unittest


import numpy

from smqtk.algorithms import get_nn_index_impls
from smqtk.algorithms.nn_index.faiss import FaissNearestNeighborsIndex
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


# Don't bother running tests of the class is not usable
if FaissNearestNeighborsIndex.is_usable():

    class TestFaissIndex (unittest.TestCase):

        def setUp(self):
            self.exhaustive_params = {
                "index_uri": None,
                "descriptor_cache_uri": None,
                "exhaustive": True,
                "index_type": "IVF100,Flat",
                "nprob": 5,
            }

            self.index_alg_params = {
                "index_uri": None,
                "descriptor_cache_uri": None,
                "exhaustive": False,
                "index_type": "IVF10,Flat",
                "nprob": 10,
            }

        def test_impl_findable(self):
            # Already here because the implementation is reporting itself as
            # usable.
            self.assertIn(FaissNearestNeighborsIndex.__name__, get_nn_index_impls())

        def test_has_model_data_no_uris(self):
            f = FaissNearestNeighborsIndex()
            self.assertFalse(f._has_model_data())

        def test_has_model_data_empty_elements(self):
            f = FaissNearestNeighborsIndex('', '', '')
            self.assertFalse(f._has_model_data())

        def test_load_faiss_model_empty_data_elements(self):
            # Construct index with valid, but empty, data URIs instances
            empty_data = 'base64://'
            f = FaissNearestNeighborsIndex(empty_data, empty_data, empty_data)
            # Load method should do nothing but set PID since given data was
            # empty.
            f._load_faiss_model()
            self.assertIsNone(f._descr_cache)
            self.assertIsNone(f._faiss_index)
            self.assertIsNotNone(f._pid)

        @mock.patch("smqtk.algorithms.nn_index.faiss"
                    ".FaissNearestNeighborsIndex._load_faiss_model")
        def test_has_model_data_valid_uris(self, m_flann_lfm):
            # Mocking flann data loading that occurs in constructor when given
            # non-empty URI targets
            f = FaissNearestNeighborsIndex(
                'base64://bW9kZWxEYXRh',  # 'modelData'
                'base64://cGFyYW1EYXRh',  # 'paramData'
                'base64://ZGVzY3JEYXRh',  # 'descrData'
            )
            self.assertTrue(f._has_model_data())

        def test_exhausive_known_descriptors_euclidean_unit(self):
            dim = 5000

            ###
            # Unit vectors -- Equal distance
            #
            index = FaissNearestNeighborsIndex(**self.exhaustive_params)
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, dtype=numpy.float32)
                v[i] = 1.
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            index.build_index(test_descriptors)
            # query descriptor -- zero vector
            # -> all modeled descriptors should be equally distance (unit
            #    corners)
            q = DescriptorMemoryElement('query', 0)
            q.set_vector(numpy.zeros(dim, dtype=numpy.float32))
            r, dists = index.nn(q, dim)
            # All dists should be 1.0, r order doesn't matter
            for d in dists:
                self.assertEqual(d, 1.)

        def test_indexingAlg_known_descriptors_euclidean_unit(self):
            dim = 5000
            k = 1000

            ###
            # Unit vectors -- Equal distance
            #
            index = FaissNearestNeighborsIndex(**self.index_alg_params)
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, dtype=numpy.float32)
                v[i] = 1.
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            index.build_index(test_descriptors)
            # query descriptor -- zero vector
            # -> all modeled descriptors should be equally distance (unit
            #    corners)
            q = DescriptorMemoryElement('query', 0)
            q.set_vector(numpy.zeros(dim, dtype=numpy.float32))
            r, dists = index.nn(q, k)
            # All dists should be 1.0, r order doesn't matter
            for d in dists:
                self.assertEqual(d, 1.)

        def test_exhausive_known_descriptors_euclidean_ordered(self):
            index = FaissNearestNeighborsIndex(**self.exhaustive_params)

            # make vectors to return in a known euclidean distance order
            dim = 1000
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, dtype=numpy.float32)
                v[i] = float(i)
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            random.shuffle(test_descriptors)
            index.build_index(test_descriptors)

            # Since descriptors were build in increasing distance from (0,0),
            # returned descriptors for a query of [0,0] should be in index
            # order.
            q = DescriptorMemoryElement('query', dim+1)
            q.set_vector(numpy.zeros(dim, dtype=numpy.float32))
            r, dists = index.nn(q, dim)
            for j, d, dist in zip(range(dim), r, dists):
                self.assertEqual(d.uuid(), j)

                v = numpy.zeros(dim, dtype=numpy.float32)
                v[j] = float(j)
                numpy.testing.assert_equal(d.vector(), v)

        def test_indexingAlg_known_descriptors_euclidean_ordered(self):
            index = FaissNearestNeighborsIndex(**self.index_alg_params)

            # make vectors to return in a known euclidean distance order
            dim = 2048
            k = 100
            test_descriptors = []
            for i in range(dim):
                v = numpy.zeros(dim, dtype=numpy.float32)
                v[i] = float(i)
                d = DescriptorMemoryElement('unit', i)
                d.set_vector(v)
                test_descriptors.append(d)
            random.shuffle(test_descriptors)
            index.build_index(test_descriptors)

            # Since descriptors were build in increasing distance from (0,0),
            # returned descriptors for a query of [0,0] should be in index
            # order.
            q = DescriptorMemoryElement('query', dim+1)
            q.set_vector(numpy.zeros(dim, dtype=numpy.float32))

            r, dists = index.nn(q, k)
            for j, d, dist in zip(range(k), r, dists):
                self.assertEqual(d.uuid(), j)

                v = numpy.zeros(dim, dtype=numpy.float32)
                v[j] = float(j)
                numpy.testing.assert_equal(d.vector(), v)

        def test_configuration(self):
            index_filepath = '/index_filepath'
            descr_cache_fp = '/descrcachefp'

            # Make configuration based on default
            c = FaissNearestNeighborsIndex.get_default_config()
            c['index_uri'] = index_filepath
            c['descriptor_cache_uri'] = descr_cache_fp
            c['exhaustive'] = False
            c['index_type'] = 'IVF100,Flat'
            c['nprob'] = 5

            # Build based on configuration
            index = FaissNearestNeighborsIndex.from_config(c)
            self.assertEqual(index._index_uri, index_filepath)
            self.assertEqual(index._descr_cache_uri, descr_cache_fp)

            c2 = index.get_config()
            self.assertEqual(c, c2)

        def test_build_index_no_descriptors(self):
            f = FaissNearestNeighborsIndex()
            self.assertRaises(
                ValueError,
                f.build_index, []
            )

        def test_build_index(self):
            # Empty memory data elements for storage
            empty_data = 'base64://'
            f = FaissNearestNeighborsIndex(empty_data, empty_data)
            # Internal elements should initialize have zero-length byte values
            self.assertEqual(len(f._index_elem.get_bytes()), 0)
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
            self.assertGreater(len(f._descr_cache_elem.get_bytes()), 0)
