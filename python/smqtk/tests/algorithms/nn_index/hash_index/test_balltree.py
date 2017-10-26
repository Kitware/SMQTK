import unittest

import mock
import numpy

from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.representation.data_element.memory_element import DataMemoryElement


class TestBallTreeHashIndex (unittest.TestCase):

    def test_is_usable(self):
        # Should always be true because major dependency (sklearn) is a package
        # requirement.
        self.assertTrue(SkLearnBallTreeHashIndex.is_usable())

    def test_default_configuration(self):
        c = SkLearnBallTreeHashIndex.get_default_config()
        self.assertEqual(len(c), 3)
        self.assertIsInstance(c['cache_element'], dict)
        self.assertIsNone(c['cache_element']['type'])
        self.assertEqual(c['leaf_size'], 40)
        self.assertIsNone(c['random_seed'])

    def test_init_without_cache(self):
        i = SkLearnBallTreeHashIndex(cache_element=None, leaf_size=52,
                                     random_seed=42)
        self.assertIsNone(i.cache_element)
        self.assertEqual(i.leaf_size, 52)
        self.assertEqual(i.random_seed, 42)
        self.assertIsNone(i.bt)

    def test_init_with_empty_cache(self):
        empty_cache = DataMemoryElement()
        i = SkLearnBallTreeHashIndex(cache_element=empty_cache,
                                     leaf_size=52,
                                     random_seed=42)
        self.assertEqual(i.cache_element, empty_cache)
        self.assertEqual(i.leaf_size, 52)
        self.assertEqual(i.random_seed, 42)
        self.assertIsNone(i.bt)

    def test_get_config(self):
        bt = SkLearnBallTreeHashIndex()
        bt_c = bt.get_config()

        self.assertEqual(len(bt_c), 3)
        self.assertIn('cache_element', bt_c)
        self.assertIn('leaf_size', bt_c)
        self.assertIn('random_seed', bt_c)

        self.assertIsInstance(bt_c['cache_element'], dict)
        self.assertIsNone(bt_c['cache_element']['type'])

    def test_init_consistency(self):
        # Test that constructing an instance with a configuration yields the
        # same config via ``get_config``.

        # - Default config should be a valid configuration for this impl.
        c = SkLearnBallTreeHashIndex.get_default_config()
        self.assertEqual(
            SkLearnBallTreeHashIndex.from_config(c).get_config(),
            c
        )
        # With non-null cache element
        c['cache_element']['type'] = 'DataMemoryElement'
        self.assertEqual(
            SkLearnBallTreeHashIndex.from_config(c).get_config(),
            c
        )

    def test_build_index_no_input(self):
        bt = SkLearnBallTreeHashIndex(random_seed=0)
        self.assertRaises(
            ValueError,
            bt.build_index, []
        )

    def test_build_index(self):
        bt = SkLearnBallTreeHashIndex(random_seed=0)
        # Make 1000 random bit vectors of length 256
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        bt.build_index(m)

        # deterministically sort index of built and source data to determine
        # that an index was built.
        self.assertIsNotNone(bt.bt)
        numpy.testing.assert_array_almost_equal(
            sorted(numpy.array(bt.bt.data).tolist()),
            sorted(m.tolist())
        )

    def test_update_index_no_input(self):
        bt = SkLearnBallTreeHashIndex(random_seed=0)
        self.assertRaises(
            ValueError,
            bt.update_index, []
        )

    def test_update_index_new_index(self):
        # Virtually the same as `test_build_index` but using update_index.
        bt = SkLearnBallTreeHashIndex(random_seed=0)
        # Make 1000 random bit vectors of length 256
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)\
                 .astype(bool)
        bt.update_index(m)

        # deterministically sort index of built and source data to determine
        # that an index was built.
        self.assertIsNotNone(bt.bt)
        numpy.testing.assert_array_almost_equal(
            sorted(numpy.array(bt.bt.data).tolist()),
            sorted(m.tolist())
        )

    def test_update_index_additive(self):
        # Test updating an existing index, i.e. rebuilding using the union of
        # previous and new data.
        bt = SkLearnBallTreeHashIndex(random_seed=0)
        # Make 1000 random bit vectors of length 256
        m1 = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256) \
                  .astype(bool)
        m2 = numpy.random.randint(0, 2, 100 * 256).reshape(100, 256) \
                  .astype(bool)

        # Build initial index
        bt.build_index(m1)
        # Current model should only contain m1's data.
        numpy.testing.assert_array_almost_equal(
            sorted(numpy.array(bt.bt.data).tolist()),
            sorted(m1.tolist())
        )

        # "Update" index with new hashes
        bt.update_index(m2)
        # New model should contain the union of the data.
        numpy.testing.assert_array_almost_equal(
            sorted(numpy.array(bt.bt.data).tolist()),
            sorted(numpy.concatenate([m1, m2], 0).tolist())
        )

    def test_count_empty(self):
        bt = SkLearnBallTreeHashIndex()
        self.assertEqual(bt.count(), 0)

    def test_count_nonempty(self):
        bt = SkLearnBallTreeHashIndex()
        # Make 1000 random bit vectors of length 256
        m = numpy.random.randint(0, 2, 234 * 256).reshape(234, 256)
        bt.build_index(m)

        self.assertEqual(bt.count(), 234)

    def test_nn_no_index(self):
        i = SkLearnBallTreeHashIndex()

        self.assertRaisesRegexp(
            ValueError,
            "No index currently set to query from",
            i.nn, [0, 0, 0]
        )

    @mock.patch('smqtk.algorithms.nn_index.hash_index.sklearn_balltree.numpy.savez')
    def test_save_model_no_cache(self, m_savez):
        bt = SkLearnBallTreeHashIndex()
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        bt.build_index(m)
        # Underlying serialization function should not have been called
        # because no cache element set.
        self.assertFalse(m_savez.called)

    def test_save_model_with_readonly_cache(self):
        cache_element = DataMemoryElement(readonly=True)
        bt = SkLearnBallTreeHashIndex(cache_element)
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        self.assertRaises(
            ValueError,
            bt.build_index, m
        )

    @mock.patch('smqtk.algorithms.nn_index.hash_index.sklearn_balltree.numpy.savez')
    def test_save_model_with_cache(self, m_savez):
        cache_element = DataMemoryElement()
        bt = SkLearnBallTreeHashIndex(cache_element, random_seed=0)
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        bt.build_index(m)
        self.assertTrue(m_savez.called)
        self.assertEqual(m_savez.call_count, 1)

    def test_load_model(self):
        # Create two index instances, building model with one, and loading
        # the other with the cache of the first instance. Each should have
        # distinct model instances, but should otherwise have equal model
        # values and parameters.
        cache_element = DataMemoryElement()
        bt1 = SkLearnBallTreeHashIndex(cache_element, random_seed=0)
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        bt1.build_index(m)

        bt2 = SkLearnBallTreeHashIndex(cache_element)
        self.assertIsNotNone(bt2.bt)

        q = numpy.random.randint(0, 2, 256).astype(bool)
        bt_neighbors, bt_dists = bt1.nn(q, 10)
        bt2_neighbors, bt2_dists = bt2.nn(q, 10)

        self.assertIsNot(bt1, bt2)
        self.assertIsNot(bt1.bt, bt2.bt)
        numpy.testing.assert_equal(bt2_neighbors, bt_neighbors)
        numpy.testing.assert_equal(bt2_dists, bt_dists)
