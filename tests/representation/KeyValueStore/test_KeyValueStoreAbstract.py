import collections
import unittest.mock as mock
import unittest

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value import KeyValueStore, NO_DEFAULT_VALUE


class DummyKVStore (KeyValueStore):

    TEST_READ_ONLY = True
    TEST_COUNT = 0

    # base-class requirements

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    # KVStore abc methods

    def __repr__(self):
        return super(DummyKVStore, self).__repr__()

    def count(self):
        return self.TEST_COUNT

    def keys(self):
        pass

    def is_read_only(self):
        return self.TEST_READ_ONLY

    def add(self, key, value):
        super(DummyKVStore, self).add(key, value)
        return self

    def add_many(self, d):
        super(DummyKVStore, self).add_many(d)
        return self

    def has(self, key):
        pass

    def get(self, key, default=NO_DEFAULT_VALUE):
        pass

    def remove(self, key):
        super(DummyKVStore, self).remove(key)
        return self

    def remove_many(self, keys):
        super(DummyKVStore, self).remove_many(keys)
        return self

    def clear(self):
        super(DummyKVStore, self).clear()
        return self


class TestKeyValueStoreAbstract (unittest.TestCase):

    def test_len(self):
        s = DummyKVStore()

        s.TEST_COUNT = 0
        assert len(s) == 0

        s.TEST_COUNT = 23456
        assert len(s) == 23456

    def test_repr(self):
        # Should return expected template string
        expected_repr = "<DummyKVStore %s>"
        actual_repr = repr(DummyKVStore())
        self.assertEqual(actual_repr, expected_repr)

    # noinspection PyUnresolvedReferences
    def test_value_iterator(self):
        expected_keys_values = {1, 5, 2345, 'foo'}

        s = DummyKVStore()
        s.keys = mock.MagicMock(return_value=expected_keys_values)
        s.get = mock.MagicMock(side_effect=lambda v: v)

        # Make sure keys now returns expected set.
        # noinspection PyTypeChecker
        # - Return value for `keys()` set above is correctly a set.
        self.assertEqual(set(s.keys()), expected_keys_values)

        # Get initial iterator. ``keys`` should have only been called once so
        # far, and ``get`` method should not have been called yet.
        # called yet.
        v_iter = s.values()
        self.assertIsInstance(v_iter, collections.abc.Iterable)
        self.assertEqual(s.keys.call_count, 1)
        self.assertEqual(s.get.call_count, 0)

        actual_values_list = set(v_iter)
        self.assertEqual(actual_values_list, expected_keys_values)
        # Keys should have been called one more time, and get should have been
        # called an equal number of times as there are keys.
        self.assertEqual(s.keys.call_count, 2)
        self.assertEqual(s.get.call_count, len(expected_keys_values))
        s.get.assert_any_call(1)
        s.get.assert_any_call(5)
        s.get.assert_any_call(2345)
        s.get.assert_any_call('foo')

    # noinspection PyUnresolvedReferences
    def test_contains(self):
        # Test that python ``has`` keyword and __contains__ method calls the
        # ``has`` method correctly.
        s = DummyKVStore()

        s.has = mock.MagicMock(return_value=True)
        self.assertTrue('some item' in s)
        s.has.assert_called_once_with('some item')

        s.has = mock.MagicMock(return_value=False)
        self.assertFalse('other item' in s)
        s.has.assert_called_once_with('other item')

    def test_get_item(self):
        s = DummyKVStore()
        s.get = mock.Mock(return_value='expected-value')
        ev = s['some-key']
        s.get.assert_called_once_with('some-key')
        self.assertEqual(ev, 'expected-value')

    def test_get_many(self):
        s = DummyKVStore()
        mock_return_values = ['expected-value', 'other-expected-value']
        s.get = mock.Mock(
            side_effect=mock_return_values
        )
        ev = list(
            s.get_many(('some-key', 'some-other-key'))
        )

        assert ev == mock_return_values

    def test_add_when_read_only(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True

        self.assertRaises(
            ReadOnlyError,
            s.add, 'k', 'v'
        )

    def test_add_when_not_read_only(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = False
        s.add('k', 'v')
        # Integer
        s.add(0, 'some value')
        # type
        s.add(object, 'some value')
        # some object instance
        s.add(object(), 'some value')

    def test_add_many_read_only(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True
        self.assertRaises(
            ReadOnlyError,
            s.add_many, {0: 1}
        )

    def test_add_many(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = False
        s.add_many({0: 1})

    def test_remove_read_only(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True
        self.assertRaises(
            ReadOnlyError,
            s.remove, 0
        )

    def test_remove(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = False
        s.remove(0)

    def test_remove_many_read_only(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True
        self.assertRaises(
            ReadOnlyError,
            s.remove_many, [0, 1]
        )

    def test_remove_many(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = False
        s.remove_many([0, 1])

    def test_clear_readonly(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True
        self.assertRaisesRegex(
            ReadOnlyError,
            "Cannot clear a read-only DummyKVStore instance.",
            s.clear
        )

    def test_clear(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = False
        s.clear()
