from __future__ import division, print_function
from math import sqrt
import unittest

import numpy
import six
from six import BytesIO

from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.utils import merge_dict


class TestItqFunctor (unittest.TestCase):

    def test_is_usable(self):
        # Should always be usable due to no non-standard dependencies.
        self.assertTrue(ItqFunctor.is_usable())

    def test_default_configuration(self):
        c = ItqFunctor.get_default_config()
        self.assertEqual(ItqFunctor.from_config(c).get_config(), c)

    def test_configuration_with_caches(self):
        expected_mean_vec = numpy.array([1, 2, 3])
        expected_rotation = numpy.eye(3)

        expected_mean_vec_bytes = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(expected_mean_vec_bytes, expected_mean_vec)
        expected_mean_vec_bytes = expected_mean_vec_bytes.getvalue()

        expected_rotation_bytes = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(expected_rotation_bytes, expected_rotation)
        expected_rotation_bytes = expected_rotation_bytes.getvalue()

        new_parts = {
            'mean_vec_cache': {
                'DataMemoryElement': {
                    'bytes': expected_mean_vec_bytes
                },
                'type': 'DataMemoryElement'
            },
            'rotation_cache': {
                'DataMemoryElement': {
                    'bytes': expected_rotation_bytes
                },
                'type': 'DataMemoryElement'
            },
            'bit_length': 153,
            'itq_iterations': 7,
            'normalize': 2,
            'random_seed': 58,
        }
        c = merge_dict(ItqFunctor.get_default_config(), new_parts)

        itq = ItqFunctor.from_config(c)

        # Checking that loaded parameters were correctly set and cache elements
        # correctly return intended vector/matrix.
        numpy.testing.assert_equal(itq.mean_vec, [1, 2, 3])
        numpy.testing.assert_equal(itq.rotation, [[1, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 1]])
        self.assertEqual(itq.bit_length, 153)
        self.assertEqual(itq.itq_iterations, 7)
        self.assertEqual(itq.normalize, 2)
        self.assertEqual(itq.random_seed, 58)

    def test_norm_vector_no_normalization(self):
        itq = ItqFunctor(normalize=None)

        v = numpy.array([0, 1])
        numpy.testing.assert_array_equal(itq._norm_vector(v), v)

        v = numpy.array([[0, 1, 1, .4, .1]])
        numpy.testing.assert_array_equal(itq._norm_vector(v), v)

        v = numpy.array([0]*128)
        numpy.testing.assert_array_equal(itq._norm_vector(v), v)

    def test_norm_vector_n2(self):
        itq = ItqFunctor(normalize=2)

        v = numpy.array([1, 0])
        numpy.testing.assert_array_almost_equal(
            itq._norm_vector(v), [1, 0]
        )

        v = numpy.array([1, 1])
        numpy.testing.assert_array_almost_equal(
            itq._norm_vector(v), [1./sqrt(2), 1./sqrt(2)]
        )

    def test_get_config_no_cache(self):
        itq = ItqFunctor(bit_length=1, itq_iterations=2, normalize=3,
                         random_seed=4)
        c = itq.get_config()
        self.assertEqual(c['bit_length'], 1)
        self.assertEqual(c['itq_iterations'], 2)
        self.assertEqual(c['normalize'], 3)
        self.assertEqual(c['random_seed'], 4)
        self.assertIsNone(c['mean_vec_cache']['type'])
        self.assertIsNone(c['rotation_cache']['type'])

    def test_get_config_with_cache_elements(self):
        itq = ItqFunctor(bit_length=5, itq_iterations=6, normalize=7,
                         random_seed=8)
        itq.mean_vec_cache_elem = DataMemoryElement('cached vec bytes')
        itq.rotation_cache_elem = DataMemoryElement('cached rot bytes')

        c = itq.get_config()
        self.assertEqual(c['bit_length'], 5)
        self.assertEqual(c['itq_iterations'], 6)
        self.assertEqual(c['normalize'], 7)
        self.assertEqual(c['random_seed'], 8)
        self.assertEqual(c['mean_vec_cache']['type'], "DataMemoryElement")
        self.assertEqual(c['mean_vec_cache']['DataMemoryElement']['bytes'],
                         'cached vec bytes')
        self.assertEqual(c['rotation_cache']['DataMemoryElement']['bytes'],
                         'cached rot bytes')

    def test_has_model(self):
        itq = ItqFunctor()
        # with no vector/rotation set, should return false.
        self.assertFalse(itq.has_model())
        # If only one of the two is None, then false should be returned.
        itq.mean_vec = 'mean vec'
        itq.rotation = None
        self.assertFalse(itq.has_model())
        itq.mean_vec = None
        itq.rotation = 'rotation'
        self.assertFalse(itq.has_model())
        # If both are not None, return true.
        itq.mean_vec = 'mean vec'
        itq.rotation = 'rotation'
        self.assertTrue(itq.has_model())

    def test_save_model_no_caches(self):
        expected_mean_vec = numpy.array([1, 2, 3])
        expected_rotation = numpy.eye(3)

        # Cache variables should remain None after save.
        itq = ItqFunctor()
        itq.mean_vec = expected_mean_vec
        itq.rotation = expected_rotation
        itq.save_model()
        self.assertIsNone(itq.mean_vec_cache_elem)
        self.assertIsNone(itq.mean_vec_cache_elem)

    def test_save_model_with_read_only_cache(self):
        # If one or both cache elements are read-only, no saving.
        expected_mean_vec = numpy.array([1, 2, 3])
        expected_rotation = numpy.eye(3)

        itq = ItqFunctor()
        itq.mean_vec = expected_mean_vec
        itq.rotation = expected_rotation

        # read-only mean-vec cache
        itq.mean_vec_cache_elem = DataMemoryElement(readonly=True)
        itq.rotation_cache_elem = DataMemoryElement(readonly=False)
        itq.save_model()
        self.assertEqual(itq.mean_vec_cache_elem.get_bytes(), six.b(''))
        self.assertEqual(itq.rotation_cache_elem.get_bytes(), six.b(''))

        # read-only rotation cache
        itq.mean_vec_cache_elem = DataMemoryElement(readonly=False)
        itq.rotation_cache_elem = DataMemoryElement(readonly=True)
        itq.save_model()
        self.assertEqual(itq.mean_vec_cache_elem.get_bytes(), six.b(''))
        self.assertEqual(itq.rotation_cache_elem.get_bytes(), six.b(''))

        # Both read-only
        itq.mean_vec_cache_elem = DataMemoryElement(readonly=True)
        itq.rotation_cache_elem = DataMemoryElement(readonly=True)
        itq.save_model()
        self.assertEqual(itq.mean_vec_cache_elem.get_bytes(), six.b(''))
        self.assertEqual(itq.rotation_cache_elem.get_bytes(), six.b(''))

    def test_save_model_with_writable_caches(self):
        # If one or both cache elements are read-only, no saving.
        expected_mean_vec = numpy.array([1, 2, 3])
        expected_rotation = numpy.eye(3)

        expected_mean_vec_bytes = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(expected_mean_vec_bytes, expected_mean_vec)
        expected_mean_vec_bytes = expected_mean_vec_bytes.getvalue()

        expected_rotation_bytes = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(expected_rotation_bytes, expected_rotation)
        expected_rotation_bytes = expected_rotation_bytes.getvalue()

        itq = ItqFunctor()
        itq.mean_vec = expected_mean_vec
        itq.rotation = expected_rotation
        itq.mean_vec_cache_elem = DataMemoryElement(readonly=False)
        itq.rotation_cache_elem = DataMemoryElement(readonly=False)

        itq.save_model()
        self.assertEqual(itq.mean_vec_cache_elem.get_bytes(),
                         expected_mean_vec_bytes)
        self.assertEqual(itq.rotation_cache_elem.get_bytes(),
                         expected_rotation_bytes)

    def test_fit_has_model(self):
        # When trying to run fit where there is already a mean vector and
        # rotation set.
        itq = ItqFunctor()
        itq.mean_vec = 'sim vec'
        itq.rotation = 'sim rot'
        self.assertRaisesRegexp(
            RuntimeError,
            "Model components have already been loaded.",
            itq.fit, []
        )

    def test_fit_short_descriptors_for_bit_length(self):
        # Should error when input descriptors have fewer dimensions than set bit
        # length for output hash codes (limitation of PCA method currently
        # used).
        fit_descriptors = []
        for i in range(3):
            d = DescriptorMemoryElement(six.b('test'), i)
            d.set_vector([-1+i, -1+i])
            fit_descriptors.append(d)

        itq = ItqFunctor(bit_length=8)
        self.assertRaisesRegexp(
            ValueError,
            "Input descriptors have fewer features than requested bit encoding",
            itq.fit, fit_descriptors
        )
        self.assertIsNone(itq.mean_vec)
        self.assertIsNone(itq.rotation)

        # Should behave the same when input is an iterable
        self.assertRaisesRegexp(
            ValueError,
            "Input descriptors have fewer features than requested bit encoding",
            itq.fit, iter(fit_descriptors)
        )
        self.assertIsNone(itq.mean_vec)
        self.assertIsNone(itq.rotation)

    def test_fit(self):
        fit_descriptors = []
        for i in range(5):
            d = DescriptorMemoryElement(six.b('test'), i)
            d.set_vector([-2. + i, -2. + i])
            fit_descriptors.append(d)

        itq = ItqFunctor(bit_length=1, random_seed=0)
        itq.fit(fit_descriptors)

        # TODO: Explanation as to why this is the expected result.
        numpy.testing.assert_array_almost_equal(itq.mean_vec, [0, 0])
        numpy.testing.assert_array_almost_equal(itq.rotation, [[1 / sqrt(2)],
                                                               [1 / sqrt(2)]])
        self.assertIsNone(itq.mean_vec_cache_elem)
        self.assertIsNone(itq.rotation_cache_elem)

    def test_fit_with_cache(self):
        fit_descriptors = []
        for i in range(5):
            d = DescriptorMemoryElement(six.b('test'), i)
            d.set_vector([-2. + i, -2. + i])
            fit_descriptors.append(d)

        itq = ItqFunctor(DataMemoryElement(), DataMemoryElement(),
                         bit_length=1, random_seed=0)
        itq.fit(fit_descriptors)

        # TODO: Explanation as to why this is the expected result.
        numpy.testing.assert_array_almost_equal(itq.mean_vec, [0, 0])
        numpy.testing.assert_array_almost_equal(itq.rotation, [[1 / sqrt(2)],
                                                               [1 / sqrt(2)]])
        self.assertIsNotNone(itq.mean_vec_cache_elem)
        numpy.testing.assert_array_almost_equal(
            numpy.load(BytesIO(itq.mean_vec_cache_elem.get_bytes())),
            [0, 0]
        )

        self.assertIsNotNone(itq.rotation_cache_elem)
        numpy.testing.assert_array_almost_equal(
            numpy.load(BytesIO(itq.rotation_cache_elem.get_bytes())),
            [[1 / sqrt(2)],
             [1 / sqrt(2)]]
        )

    def test_get_hash(self):
        fit_descriptors = []
        for i in range(5):
            d = DescriptorMemoryElement(six.b('test'), i)
            d.set_vector([-2. + i, -2. + i])
            fit_descriptors.append(d)

        # The following "rotation" matrix should cause any 2-feature descriptor
        # to the right of the line ``y = -x`` to be True, and to the left as
        # False. If on the line, should be True.
        itq = ItqFunctor(bit_length=1, random_seed=0)
        itq.mean_vec = numpy.array([0., 0.])
        itq.rotation = numpy.array([[1. / sqrt(2)],
                                    [1. / sqrt(2)]])

        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([1, 1])), [True])
        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([-1, -1])), [False])

        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([-1, 1])), [True])
        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([-1.001, 1])), [False])
        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([-1, 1.001])), [True])

        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([1, -1])), [True])
        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([1, -1.001])), [False])
        numpy.testing.assert_array_equal(
            itq.get_hash(numpy.array([1.001, -1])), [True])
