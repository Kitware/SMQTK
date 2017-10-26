import unittest

import numpy

from smqtk.algorithms.nn_index.hash_index.linear import LinearHashIndex
from smqtk.representation.data_element.memory_element import DataMemoryElement


class TestLinearHashIndex (unittest.TestCase):

    def test_is_usable(self):
        # Should always be true since this impl does no have special deps.
        self.assertTrue(LinearHashIndex.is_usable())

    def test_default_config(self):
        c = LinearHashIndex.get_default_config()
        self.assertEqual(len(c), 1)
        self.assertIsNone(c['cache_element']['type'])

    def test_from_config_no_cache(self):
        # Default config is valid and specifies no cache.
        c = LinearHashIndex.get_default_config()
        i = LinearHashIndex.from_config(c)
        self.assertIsNone(i.cache_element)
        self.assertEqual(i.index, set())

    def test_from_config_with_cache(self):
        c = LinearHashIndex.get_default_config()
        c['cache_element']['type'] = "DataMemoryElement"
        i = LinearHashIndex.from_config(c)
        self.assertIsInstance(i.cache_element, DataMemoryElement)
        self.assertEqual(i.index, set())

    def test_get_config(self):
        i = LinearHashIndex()

        # Without cache element
        expected_c = LinearHashIndex.get_default_config()
        self.assertEqual(i.get_config(), expected_c)

        # With cache element
        i.cache_element = DataMemoryElement()
        expected_c['cache_element']['type'] = 'DataMemoryElement'
        self.assertEqual(i.get_config(), expected_c)

    def test_build_index_no_cache(self):
        i = LinearHashIndex()
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 1],
                       [0, 0, 1]])
        self.assertEqual(i.index, {1, 2, 3, 4})
        self.assertIsNone(i.cache_element)

    def test_build_index_with_cache(self):
        cache_element = DataMemoryElement()
        i = LinearHashIndex(cache_element)
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 1],
                       [0, 0, 1]])
        self.assertEqual(i.index, {1, 2, 3, 4})
        self.assertFalse(cache_element.is_empty())

    def test_build_index_no_input(self):
        i = LinearHashIndex()
        self.assertRaises(
            ValueError,
            i.build_index, []
        )

    def test_nn(self):
        i = LinearHashIndex()
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],
                       [1, 1, 0],
                       [0, 1, 1],
                       [0, 0, 1]])
        near_codes, near_dists = i.nn([0, 0, 0], 4)
        self.assertEqual(set(map(tuple, near_codes[:2])),
                         {(0, 1, 0), (0, 0, 1)})
        self.assertEqual(set(map(tuple, near_codes[2:])),
                         {(1, 1, 0), (0, 1, 1)})
        numpy.testing.assert_array_almost_equal(near_dists,
                                                (1/3., 1/3., 2/3., 2/3.))

    def test_save_cache(self):
        cache_element = DataMemoryElement()
        self.assertTrue(cache_element.is_empty())

        i = LinearHashIndex(cache_element)
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 1],
                       [0, 0, 1]])
        self.assertFalse(cache_element.is_empty())
        self.assertTrue(len(cache_element.get_bytes()) > 0)

    def test_save_cache_readonly(self):
        ro_cache = DataMemoryElement(readonly=True)
        i = LinearHashIndex(ro_cache)
        self.assertRaisesRegexp(
            ValueError,
            "is read-only",
            i.build_index,
            [[0, 1, 0],
             [1, 0, 0],
             [0, 1, 1],
             [0, 0, 1]]
        )

    def test_load_cache(self):
        cache_element = DataMemoryElement()
        i1 = LinearHashIndex(cache_element)
        # noinspection PyTypeChecker
        i1.build_index([[0, 1, 0],
                        [1, 0, 0],
                        [0, 1, 1],
                        [0, 0, 1]])

        # load called on initialization.
        i2 = LinearHashIndex(cache_element)

        self.assertEqual(i1.cache_element, i2.cache_element)
        self.assertEqual(i1.index, i2.index)
