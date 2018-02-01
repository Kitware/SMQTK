import collections
import mock
import unittest

import nose.tools

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value import KeyValueStore, NO_DEFAULT_VALUE, \
    get_key_value_store_impls
from smqtk.representation.key_value.memory import MemoryKeyValueStore


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

    def clear(self):
        super(DummyKVStore, self).clear()


class TestKeyValueStoreAbstract (unittest.TestCase):

    def test_repr(self):
        # Should return expected template string
        expected_repr = "<DummyKVStore %s>"
        actual_repr = repr(DummyKVStore())
        nose.tools.assert_equal(actual_repr, expected_repr)

    def test_add_when_read_only(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True

        nose.tools.assert_raises(
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

    def test_add_not_hashable(self):
        s = DummyKVStore()
        # list
        nose.tools.assert_raises(
            ValueError,
            s.add, [1, 2], 'some value'
        )
        # set
        nose.tools.assert_raises(
            ValueError,
            s.add, {1, 2}, 'some value'
        )

    def test_len(self):
        s = DummyKVStore()

        s.TEST_COUNT = 0
        assert len(s) == 0

        s.TEST_COUNT = 23456
        assert len(s) == 23456

    # noinspection PyUnresolvedReferences
    def test_value_iterator(self):
        expected_keys_values = {1, 5, 2345, 'foo'}

        s = DummyKVStore()
        s.keys = mock.MagicMock(return_value=expected_keys_values)
        s.get = mock.MagicMock(side_effect=lambda v: v)

        # Make sure keys now returns expected list.
        nose.tools.assert_equal(s.keys(), expected_keys_values)

        # Get initial iterator. ``keys`` should have only been called once so
        # far, and ``get`` method should not have been called yet.
        # called yet.
        v_iter = s.values()
        nose.tools.assert_is_instance(v_iter, collections.Iterable)
        nose.tools.assert_equal(s.keys.call_count, 1)
        nose.tools.assert_equal(s.get.call_count, 0)

        actual_values_list = set(v_iter)
        nose.tools.assert_equal(actual_values_list, expected_keys_values)
        # Keys should have been called one more time, and get should have been
        # called an equal number of times as there are keys.
        nose.tools.assert_equal(s.keys.call_count, 2)
        nose.tools.assert_equal(s.get.call_count, len(expected_keys_values))
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
        nose.tools.assert_true('some item' in s)
        s.has.assert_called_once_with('some item')

        s.has = mock.MagicMock(return_value=False)
        nose.tools.assert_false('other item' in s)
        s.has.assert_called_once_with('other item')

    def test_clear_base(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = False
        s.clear()

    def test_clear_readonly(self):
        s = DummyKVStore()
        s.TEST_READ_ONLY = True
        nose.tools.assert_raises_regexp(
            ReadOnlyError,
            "Cannot clear a read-only DummyKVStore instance.",
            s.clear
        )


def test_kvstore_impl_getter():
    # At least the in-memory implementation should always be available, so make
    # sure at least that is returned from the getter.
    d = get_key_value_store_impls()
    nose.tools.assert_is_instance(d, dict)
    nose.tools.assert_in('MemoryKeyValueStore', d)
    nose.tools.assert_equal(d['MemoryKeyValueStore'], MemoryKeyValueStore)
