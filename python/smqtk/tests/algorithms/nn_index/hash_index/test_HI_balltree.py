import unittest

import mock
import nose.tools
import numpy

from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.representation.data_element.memory_element import DataMemoryElement


class TestBallTreeHashIndex (unittest.TestCase):

    def test_is_usable(self):
        # Should always be true because major dependency (sklearn) is a package
        # requirement.
        nose.tools.assert_true(SkLearnBallTreeHashIndex.is_usable())

    def test_default_configuration(self):
        c = SkLearnBallTreeHashIndex.get_default_config()
        nose.tools.assert_equal(len(c), 3)
        nose.tools.assert_is_instance(c['cache_element'], dict)
        nose.tools.assert_is_none(c['cache_element']['type'])
        nose.tools.assert_equal(c['leaf_size'], 40)
        nose.tools.assert_is_none(c['random_seed'])

    def test_init_without_cache(self):
        i = SkLearnBallTreeHashIndex(cache_element=None, leaf_size=52,
                                     random_seed=42)
        nose.tools.assert_is_none(i.cache_element)
        nose.tools.assert_equal(i.leaf_size, 52)
        nose.tools.assert_equal(i.random_seed, 42)
        nose.tools.assert_is_none(i.bt)

    def test_init_with_empty_cache(self):
        empty_cache = DataMemoryElement()
        i = SkLearnBallTreeHashIndex(cache_element=empty_cache,
                                     leaf_size=52,
                                     random_seed=42)
        nose.tools.assert_equal(i.cache_element, empty_cache)
        nose.tools.assert_equal(i.leaf_size, 52)
        nose.tools.assert_equal(i.random_seed, 42)
        nose.tools.assert_is_none(i.bt)

    def test_get_config(self):
        bt = SkLearnBallTreeHashIndex()
        bt_c = bt.get_config()

        nose.tools.assert_equal(len(bt_c), 3)
        nose.tools.assert_in('cache_element', bt_c)
        nose.tools.assert_in('leaf_size', bt_c)
        nose.tools.assert_in('random_seed', bt_c)

        nose.tools.assert_is_instance(bt_c['cache_element'], dict)
        nose.tools.assert_is_none(bt_c['cache_element']['type'])

    def test_init_consistency(self):
        # Test that constructing an instance with a configuration yields the
        # same config via ``get_config``.

        # - Default config should be a valid configuration for this impl.
        c = SkLearnBallTreeHashIndex.get_default_config()
        nose.tools.assert_equal(
            SkLearnBallTreeHashIndex.from_config(c).get_config(),
            c
        )
        # With non-null cache element
        c['cache_element']['type'] = 'DataMemoryElement'
        nose.tools.assert_equal(
            SkLearnBallTreeHashIndex.from_config(c).get_config(),
            c
        )

    def test_invalid_build(self):
        bt = SkLearnBallTreeHashIndex()
        nose.tools.assert_raises(
            ValueError,
            bt.build_index,
            []
        )

    def test_build_index(self):
        bt = SkLearnBallTreeHashIndex(random_seed=0)
        # Make 1000 random bit vectors of length 256
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        bt.build_index(m)

        # deterministically sort index of built and source data to determine
        # that an index was built.
        nose.tools.assert_is_not_none(bt.bt)
        numpy.testing.assert_array_almost_equal(
            sorted(numpy.array(bt.bt.data).tolist()),
            sorted(m.tolist())
        )

    def test_count_empty(self):
        bt = SkLearnBallTreeHashIndex()
        nose.tools.assert_equal(bt.count(), 0)

    def test_count_nonempty(self):
        bt = SkLearnBallTreeHashIndex()
        # Make 1000 random bit vectors of length 256
        m = numpy.random.randint(0, 2, 234 * 256).reshape(234, 256)
        bt.build_index(m)

        nose.tools.assert_equal(bt.count(), 234)

    def test_nn_no_index(self):
        i = SkLearnBallTreeHashIndex()

        nose.tools.assert_raises_regexp(
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
        nose.tools.assert_false(m_savez.called)

    def test_save_model_with_readonly_cache(self):
        cache_element = DataMemoryElement(readonly=True)
        bt = SkLearnBallTreeHashIndex(cache_element)
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        nose.tools.assert_raises(
            ValueError,
            bt.build_index, m
        )

    @mock.patch('smqtk.algorithms.nn_index.hash_index.sklearn_balltree.numpy.savez')
    def test_save_model_with_cache(self, m_savez):
        cache_element = DataMemoryElement()
        bt = SkLearnBallTreeHashIndex(cache_element, random_seed=0)
        m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
        bt.build_index(m)
        nose.tools.assert_true(m_savez.called)
        nose.tools.assert_equal(m_savez.call_count, 1)

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
        nose.tools.assert_is_not_none(bt2.bt)

        q = numpy.random.randint(0, 2, 256).astype(bool)
        bt_neighbors, bt_dists = bt1.nn(q, 10)
        bt2_neighbors, bt2_dists = bt2.nn(q, 10)

        nose.tools.assert_is_not(bt1, bt2)
        nose.tools.assert_is_not(bt1.bt, bt2.bt)
        numpy.testing.assert_equal(bt2_neighbors, bt_neighbors)
        numpy.testing.assert_equal(bt2_dists, bt_dists)
