import mock
import unittest

import six

from smqtk.algorithms.nn_index.hash_index import HashIndex, get_hash_index_impls


class DummyHI (HashIndex):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def count(self):
        return 0

    def build_index(self, hashes):
        return super(DummyHI, self).build_index(hashes)

    def update_index(self, hashes):
        return super(DummyHI, self).update_index(hashes)

    def nn(self, h, n=1):
        return super(DummyHI, self).nn(h, n)


class TestHashIndex (unittest.TestCase):

    def test_get_impls(self):
        m = get_hash_index_impls()
        self.assertIsInstance(m, dict)
        for cls in six.itervalues(m):
            self.assertTrue(issubclass(cls, HashIndex))

    def test_build_index_empty_iter(self):
        idx = DummyHI()
        self.assertRaisesRegexp(
            ValueError,
            "No hash vectors.*",
            idx.build_index, []
        )

    def test_build_index_with_values(self):
        idx = DummyHI()
        # No error should be returned. Returned iterable contents should match
        # input values.
        ret = idx.build_index([0, 1, 2])
        self.assertSetEqual(
            set(ret),
            {0, 1, 2}
        )

    def test_update_index_empty_iter(self):
        idx = DummyHI()
        self.assertRaisesRegexp(
            ValueError,
            "No hash vectors.*",
            idx.update_index, []
        )

    def test_update_index_with_values(self):
        idx = DummyHI()
        # No error should be returned. Returned iterable contents should match
        # input values.
        ret = idx.update_index([0, 1, 2])
        self.assertSetEqual(
            set(ret),
            {0, 1, 2}
        )

    def test_nn_no_index(self):
        idx = DummyHI()
        self.assertRaises(
            ValueError,
            idx.nn, 'something'
        )

    def test_nn_has_count(self):
        idx = DummyHI()
        idx.count = mock.MagicMock()
        idx.count.return_value = 10
        # This call should now pass that count returns something greater than 0.
        idx.nn('dummy')
