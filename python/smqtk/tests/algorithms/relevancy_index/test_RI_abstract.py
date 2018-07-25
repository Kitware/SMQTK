from __future__ import division, print_function
import unittest

import mock

from smqtk.algorithms.relevancy_index import RelevancyIndex


class DummyRI (RelevancyIndex):

    @classmethod
    def is_usable(cls):
        return True

    def rank(self, pos, neg):
        pass

    def get_config(self):
        pass

    def count(self):
        return 0

    def build_index(self, descriptors):
        pass


class TestSimilarityIndexAbstract (unittest.TestCase):

    def test_count(self):
        index = DummyRI()
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.count(), len(index))

        # Pretend that there were things in there. Len should pass it though
        index.count = mock.Mock()
        index.count.return_value = 5
        self.assertEqual(len(index), 5)
