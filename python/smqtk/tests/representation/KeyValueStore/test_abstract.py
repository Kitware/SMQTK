import unittest

import nose.tools

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value import KeyValueStore, NO_DEFAULT_VALUE


class DummyKVStore (KeyValueStore):

    TEST_READ_ONLY = True

    # base-class requirements

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    # KVStore abc methods

    def __repr__(self):
        return super(DummyKVStore, self).__repr__()

    def is_read_only(self):
        return self.TEST_READ_ONLY

    def add(self, key, value):
        super(DummyKVStore, self).add(key, value)
        return self

    def has(self, key):
        pass

    def get(self, key, default=NO_DEFAULT_VALUE):
        pass


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

    def test_add_not_string(self):
        s = DummyKVStore()
        # Integer
        nose.tools.assert_raises(
            ValueError,
            s.add, 0, 'some value'
        )
        # type
        nose.tools.assert_raises(
            ValueError,
            s.add, object, 'some value'
        )
        # some object instance
        nose.tools.assert_raises(
            ValueError,
            s.add, object(), 'some value'
        )
