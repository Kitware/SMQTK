from __future__ import division, print_function
import unittest

import numpy
from six import BytesIO

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

    def test_update_index_no_input(self):
        i = LinearHashIndex()
        self.assertRaises(
            ValueError,
            i.update_index, []
        )

    def test_update_index_no_index(self):
        # Test calling update index with no existing index.  Should result the
        # same as calling build_index with no index.
        i = LinearHashIndex()
        # noinspection PyTypeChecker
        i.update_index([[0, 1, 0],
                        [1, 0, 0],
                        [0, 1, 1],
                        [0, 0, 1]])
        self.assertEqual(i.index, {1, 2, 3, 4})
        self.assertIsNone(i.cache_element)

    def test_update_index_add_hashes(self):
        i = LinearHashIndex()
        # Build index with some initial hashes
        # noinspection PyTypeChecker
        i.build_index([[0, 0],
                       [0, 1]])
        self.assertSetEqual(i.index, {0, 1})
        # Update index with new stuff
        # noinspection PyTypeChecker
        i.update_index([[1, 0],
                        [1, 1]])
        self.assertSetEqual(i.index, {0, 1, 2, 3})

    def test_remove_from_index_single_not_in_index(self):
        # Test attempting to remove single hash not in the index.
        i = LinearHashIndex()
        i.index = {0, 1, 2}
        self.assertRaises(
            KeyError,
            i.remove_from_index,
            [[1, 0, 0]]  # 4
        )
        self.assertSetEqual(i.index, {0, 1, 2})

    def test_remove_from_index_one_of_many_not_in_index(self):
        # Test attempting to remove hashes where one of them is not in the
        # index.
        i = LinearHashIndex()
        i.index = {0, 1, 2}
        self.assertRaises(
            KeyError,
            i.remove_from_index, [[0, 0],  # 0
                                  [0, 1],  # 1
                                  [1, 1]]  # 3
        )
        # Check that the index has not been modified.
        self.assertSetEqual(i.index, {0, 1, 2})

    def test_remove_from_index(self):
        # Test that actual removal occurs.
        i = LinearHashIndex()
        i.index = {0, 1, 2}
        # noinspection PyTypeChecker
        i.remove_from_index([[0, 0],
                             [1, 0]])
        self.assertSetEqual(i.index, {1})

    def test_nn(self):
        i = LinearHashIndex()
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],
                       [1, 1, 0],
                       [0, 1, 1],
                       [0, 0, 1]])
        # noinspection PyTypeChecker
        near_codes, near_dists = i.nn([0, 0, 0], 4)
        self.assertEqual(set(map(tuple, near_codes[:2])),
                         {(0, 1, 0), (0, 0, 1)})
        self.assertEqual(set(map(tuple, near_codes[2:])),
                         {(1, 1, 0), (0, 1, 1)})
        numpy.testing.assert_array_almost_equal(near_dists,
                                                (1/3., 1/3., 2/3., 2/3.))

    def test_save_cache_build_index(self):
        cache_element = DataMemoryElement()
        self.assertTrue(cache_element.is_empty())

        i = LinearHashIndex(cache_element)
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 1],
                       [0, 0, 1]])
        self.assertFalse(cache_element.is_empty())
        # Check byte content
        expected_cache = {1, 2, 3, 4}
        actual_cache = set(numpy.load(BytesIO(cache_element.get_bytes())))
        self.assertSetEqual(expected_cache, actual_cache)

    def test_save_cache_update_index(self):
        cache_element = DataMemoryElement()
        self.assertTrue(cache_element.is_empty())

        i = LinearHashIndex(cache_element)
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],   # 2
                       [1, 0, 0]])  # 4
        # noinspection PyTypeChecker
        i.update_index([[0, 1, 1],   # 3
                        [0, 0, 1]])  # 1
        self.assertFalse(cache_element.is_empty())
        # Check byte content
        expected_cache = {1, 2, 3, 4}
        actual_cache = set(numpy.load(BytesIO(cache_element.get_bytes())))
        self.assertSetEqual(expected_cache, actual_cache)

    def test_save_cache_remove_from_index(self):
        # Test that the cache is updated appropriately on a removal.
        cache_element = DataMemoryElement()
        self.assertTrue(cache_element.is_empty())

        i = LinearHashIndex(cache_element)
        # noinspection PyTypeChecker
        i.build_index([[0, 1, 0],   # 2
                       [0, 1, 1],   # 3
                       [1, 0, 0],   # 4
                       [1, 1, 0]])  # 6
        self.assertFalse(cache_element.is_empty())
        self.assertSetEqual(
            set(numpy.load(BytesIO(cache_element.get_bytes()))),
            {2, 3, 4, 6}
        )

        # noinspection PyTypeChecker
        i.remove_from_index([[0, 1, 1],   # 3
                             [1, 0, 0]])  # 4
        self.assertFalse(cache_element.is_empty())
        self.assertSetEqual(
            set(numpy.load(BytesIO(cache_element.get_bytes()))),
            {2, 6}
        )

    def test_save_cache_readonly_build_index(self):
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

    def test_save_cache_readonly_update_index(self):
        ro_cache = DataMemoryElement(readonly=True)
        i = LinearHashIndex(ro_cache)
        self.assertRaisesRegexp(
            ValueError,
            "is read-only",
            i.update_index,
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
