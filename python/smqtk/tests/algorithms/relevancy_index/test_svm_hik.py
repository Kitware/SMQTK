import unittest

import nose.tools as ntools
import numpy as np

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms.relevancy_index.libsvm_hik import LibSvmHikRelevancyIndex


__author__ = "paul.tunison@kitware.com"


if LibSvmHikRelevancyIndex.is_usable():

    class TestIqrSvmHik (unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            # Don't need to clear cache because we're setting the vectors here
            cls.d0 = DescriptorMemoryElement('index', 0)
            cls.d0.set_vector(np.array([1, 0, 0, 0, 0], float))
            cls.d1 = DescriptorMemoryElement('index', 1)
            cls.d1.set_vector(np.array([0, 1, 0, 0, 0], float))
            cls.d2 = DescriptorMemoryElement('index', 2)
            cls.d2.set_vector(np.array([0, 0, 1, 0, 0], float))
            cls.d3 = DescriptorMemoryElement('index', 3)
            cls.d3.set_vector(np.array([0, 0, 0, 1, 0], float))
            cls.d4 = DescriptorMemoryElement('index', 4)
            cls.d4.set_vector(np.array([0, 0, 0, 0, 1], float))
            cls.d5 = DescriptorMemoryElement('index', 5)
            cls.d5.set_vector(np.array([0.5, 0, 0.5, 0, 0], float))
            cls.d6 = DescriptorMemoryElement('index', 6)
            cls.d6.set_vector(np.array([.2, .2, .2, .2, .2], float))
            cls.index_descriptors = [cls.d0, cls.d1, cls.d2, cls.d3, cls.d4,
                                     cls.d5, cls.d6]

            cls.q_pos = DescriptorMemoryElement('query', 0)
            cls.q_pos.set_vector(np.array([.75, .25, 0, 0,  0], float))
            cls.q_neg = DescriptorMemoryElement('query', 1)
            cls.q_neg.set_vector(np.array([0,   0,   0, .5, .5], float))

        def test_configuration(self):
            c = LibSvmHikRelevancyIndex.get_default_config()
            ntools.assert_in('descr_cache_filepath', c)

            # change default for something different
            c['descr_cache_filepath'] = 'foobar.thing'

            iqr_index = LibSvmHikRelevancyIndex.from_config(c)
            ntools.assert_equal(iqr_index.descr_cache_fp,
                                c['descr_cache_filepath'])

            # test config idempotency
            ntools.assert_dict_equal(c, iqr_index.get_config())

        def test_rank_no_neg(self):
            iqr_index = LibSvmHikRelevancyIndex()
            iqr_index.build_index(self.index_descriptors)
            # index should auto-select some negative examples, thus not raising
            # an exception.
            iqr_index.rank([self.q_pos], [])

        def test_rank_no_pos(self):
            iqr_index = LibSvmHikRelevancyIndex()
            iqr_index.build_index(self.index_descriptors)
            ntools.assert_raises(ValueError, iqr_index.rank, [], [self.q_neg])

        def test_rank_no_input(self):
            iqr_index = LibSvmHikRelevancyIndex()
            iqr_index.build_index(self.index_descriptors)
            ntools.assert_raises(ValueError, iqr_index.rank, [], [])

        def test_count(self):
            iqr_index = LibSvmHikRelevancyIndex()
            ntools.assert_equal(iqr_index.count(), 0)
            iqr_index.build_index(self.index_descriptors)
            ntools.assert_equal(iqr_index.count(), 7)

        def test_simple_iqr_scenario(self):
            # Make some descriptors;
            # Pick some from created set that are close to each other and use as
            #   positive query, picking some other random descriptors as
            #   negative examples.
            # Rank index based on chosen pos/neg
            # Check that positive choices are at the top of the ranking (closest
            #   to 0) and negative choices are closest to the bottom.
            iqr_index = LibSvmHikRelevancyIndex()
            iqr_index.build_index(self.index_descriptors)

            rank = iqr_index.rank([self.q_pos], [self.q_neg])
            rank_ordered = sorted(rank.items(), key=lambda e: e[1], reverse=True)

            # Check expected ordering
            # 0-5-1-2-6-3-4
            # - 2 should end up coming before 6, because 6 has more intersection
            #   with the negative example.
            ntools.assert_equal(rank_ordered[0][0], self.d0)
            ntools.assert_equal(rank_ordered[1][0], self.d5)
            ntools.assert_equal(rank_ordered[2][0], self.d1)
            ntools.assert_equal(rank_ordered[3][0], self.d2)
            ntools.assert_equal(rank_ordered[4][0], self.d6)
            ntools.assert_equal(rank_ordered[5][0], self.d3)
            ntools.assert_equal(rank_ordered[6][0], self.d4)
