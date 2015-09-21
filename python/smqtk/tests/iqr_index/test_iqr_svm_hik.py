import unittest

import nose.tools as ntools
import numpy as np

from smqtk.data_rep.descriptor_element_impl.local_elements import \
    DescriptorMemoryElement
from smqtk.iqr_index.libsvm_hik import LibSvmHikIqrIndex


__author__ = 'purg'


d0 = DescriptorMemoryElement('index', 0)
d0.set_vector(np.array([1, 0, 0, 0, 0], float))
d1 = DescriptorMemoryElement('index', 1)
d1.set_vector(np.array([0, 1, 0, 0, 0], float))
d2 = DescriptorMemoryElement('index', 2)
d2.set_vector(np.array([0, 0, 1, 0, 0], float))
d3 = DescriptorMemoryElement('index', 3)
d3.set_vector(np.array([0, 0, 0, 1, 0], float))
d4 = DescriptorMemoryElement('index', 4)
d4.set_vector(np.array([0, 0, 0, 0, 1], float))
d5 = DescriptorMemoryElement('index', 5)
d5.set_vector(np.array([0.5, 0, 0.5, 0, 0], float))
d6 = DescriptorMemoryElement('index', 6)
d6.set_vector(np.array([.2, .2, .2, .2, .2], float))
index_descriptors = [d0, d1, d2, d3, d4, d5, d6]

q_pos = DescriptorMemoryElement('query', 0)
q_pos.set_vector(np.array([.75, .25, 0, 0, 0], float))
q_neg = DescriptorMemoryElement('query', 1)
q_neg.set_vector(np.array([0, 0, 0, .5, .5], float))


class TestIqrSvmHik (unittest.TestCase):

    def test_configuration(self):
        c = LibSvmHikIqrIndex.default_config()
        ntools.assert_in('descr_cache_filepath', c)

        # change default for something different
        c['descr_cache_filepath'] = 'foobar.thing'

        iqr_index = LibSvmHikIqrIndex.from_config(c)
        ntools.assert_equal(iqr_index._descr_cache_fp,
                            c['descr_cache_filepath'])

        # test config idempotency
        ntools.assert_dict_equal(c, iqr_index.get_config())

    def test_rank_no_neg(self):
        iqr_index = LibSvmHikIqrIndex()
        iqr_index.build_index(index_descriptors)
        ntools.assert_raises(ValueError, iqr_index.rank, [q_pos])

    def test_rank_no_pos(self):
        iqr_index = LibSvmHikIqrIndex()
        iqr_index.build_index(index_descriptors)
        ntools.assert_raises(ValueError, iqr_index.rank, [], [q_neg])

    def test_rank_no_input(self):
        iqr_index = LibSvmHikIqrIndex()
        iqr_index.build_index(index_descriptors)
        ntools.assert_raises(ValueError, iqr_index.rank, [])

    def test_count(self):
        iqr_index = LibSvmHikIqrIndex()
        ntools.assert_equal(iqr_index.count(), 0)
        iqr_index.build_index(index_descriptors)
        ntools.assert_equal(iqr_index.count(), 7)

    def test_simple_iqr_scenario(self):
        # Make some descriptors;
        # Pick some from created set that are close to each other and use as
        #   positive query, picking some other random descriptors as negative
        #   examples.
        # Rank index based on chosen pos/neg
        # Check that positive choices are at the top of the ranking (closest to
        #   0) and negative choices are closest to the bottom.
        iqr_index = LibSvmHikIqrIndex()
        iqr_index.build_index(index_descriptors)

        rank = iqr_index.rank([q_pos], [q_neg])
        rank_ordered = sorted(rank.items(), key=lambda e: e[1], reverse=True)

        # Check expected ordering
        # 0-5-1-2-6-3-4
        # - 2 should end up coming before 6, because 6 has more intersection
        #   with the negative example.
        ntools.assert_equal(rank_ordered[0][0], d0)
        ntools.assert_equal(rank_ordered[1][0], d5)
        ntools.assert_equal(rank_ordered[2][0], d1)
        ntools.assert_equal(rank_ordered[3][0], d2)
        ntools.assert_equal(rank_ordered[4][0], d6)
        ntools.assert_equal(rank_ordered[5][0], d3)
        ntools.assert_equal(rank_ordered[6][0], d4)
