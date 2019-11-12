import unittest

import mock
import six
from six.moves import cPickle as pickle

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.data_element.memory_element import (
    BYTES_CONFIG_ENCODING,
    DataMemoryElement,
)
from smqtk.representation.key_value.memory import MemoryKeyValueStore


class TestMemoryKeyValueStore (unittest.TestCase):

    def test_is_usable(self):
        # Should always be usable
        self.assertTrue(MemoryKeyValueStore.is_usable())

    def test_get_default(self):
        # Check default config
        default_config = MemoryKeyValueStore.get_default_config()
        self.assertIsInstance(default_config, dict)
        # - Should just contain cache element property, which is a nested
        #   plugin config with no default type.
        self.assertIn('cache_element', default_config)
        self.assertIn('type', default_config['cache_element'])
        self.assertIsNone(default_config['cache_element']['type'])

    def test_from_config_empty_config(self):
        # should resort to default parameters
        s = MemoryKeyValueStore.from_config({})
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, {})

    def test_from_config_none_value(self):
        # When cache_element value is None
        expected_config = {'cache_element': None}
        s = MemoryKeyValueStore.from_config(expected_config)
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, {})

    def test_from_config_none_type(self):
        # When config map given, but plugin type set to null/None
        config = {'cache_element': {
            'some_type': {'param': None},
            'type': None,
        }}
        s = MemoryKeyValueStore.from_config(config)
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, {})

    def test_from_config_with_cache_element(self):
        # Pickled dictionary with a known entry
        expected_table = {'some_key': 'some_value'}
        empty_dict_pickle_str = \
            pickle.dumps(expected_table).decode(BYTES_CONFIG_ENCODING)

        # Test construction with memory data element.
        config = {'cache_element': {
            'DataMemoryElement': {
                'bytes': empty_dict_pickle_str,
            },
            'type': 'DataMemoryElement'
        }}
        s = MemoryKeyValueStore.from_config(config)
        self.assertIsInstance(s._cache_element, DataMemoryElement)
        self.assertEqual(s._table, expected_table)

    def test_new_no_cache(self):
        """ Test construction with no cache element. """
        s = MemoryKeyValueStore()
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, {})

    def test_new_empty_cache(self):
        """
        Test construction with a cache element set with no bytes (empty).
        """
        c = DataMemoryElement()
        s = MemoryKeyValueStore(c)
        self.assertEqual(s._cache_element, c)
        self.assertEqual(s._table, {})

    def test_new_cached_table(self):
        """
        Test construction with a cached table.
        """
        expected_table = {
            'a': 'b',
            'c': 1,
            'asdfghsdfg': None,
            'r3adf3a#+': [4, 5, 6, '7'],
        }
        expected_table_pickle = pickle.dumps(expected_table, 2)

        c = DataMemoryElement(expected_table_pickle)
        s = MemoryKeyValueStore(c)
        self.assertEqual(s._cache_element, c)
        self.assertEqual(s._table, expected_table)

    def test_repr_no_cache(self):
        """
        Test representational string when no cache is set.
        """
        expected_repr = '<MemoryKeyValueStore cache_element: None>'
        s = MemoryKeyValueStore()
        actual_repr = repr(s)
        self.assertEqual(actual_repr, expected_repr)

    def test_repr_simple_cache(self):
        """
        Test representational string when a cache element is set.
        """
        c = DataMemoryElement()
        s = MemoryKeyValueStore(c)
        expected_repr = "<MemoryKeyValueStore cache_element: " \
                        "DataMemoryElement{len(bytes): 0, content_type: " \
                        "None, readonly: False}>"
        self.assertEqual(repr(s), expected_repr)

    def test_count(self):
        """
        Test that count returns appropriately based on table state.
        """
        s = MemoryKeyValueStore()
        assert s.count() == 0
        s._table = {
            0: 0,
            1: 1,
            'a': True,
            None: False
        }
        assert s.count() == 4

    def test_get_config_no_cache_elem(self):
        """
        Test that configuration returned reflects no cache element being set.
        """
        s = MemoryKeyValueStore()
        s._cache_element = None
        # We expect an default DataElement config (no impl type defined)
        c = s.get_config()
        self.assertIn('cache_element', c)
        self.assertIsNone(c['cache_element']['type'])

    def test_get_config_mem_cache_elem(self):
        """
        Test that configuration returned reflects the cache element that is
        set.
        """
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement(b'someBytes', 'text/plain',
                                             readonly=False)
        expected_config = {'cache_element': {
            "DataMemoryElement": {
                'bytes': 'someBytes',
                'content_type': 'text/plain',
                'readonly': False,
            },
            'type': 'DataMemoryElement'
        }}
        self.assertEqual(s.get_config(), expected_config)

    def test_keys_empty(self):
        """
        Test that no keys are present in a freshly constructed instance.
        """
        s = MemoryKeyValueStore()
        self.assertEqual(list(s.keys()), [])

    def test_keys_with_table(self):
        """
        Test that keys returned reflect the table state.
        """
        s = MemoryKeyValueStore()
        s._table = {
            'a': 'b',
            'c': 1,
            'asdfghsdfg': None,
            'r3adf3a#+': [4, 5, 6, '7'],
        }
        self.assertSetEqual(
            set(s.keys()),
            {'a', 'c', 'asdfghsdfg', 'r3adf3a#+'}
        )

    def test_read_only_no_cache(self):
        """ Test that read_only is false when there is no cache. """
        s = MemoryKeyValueStore()
        self.assertIsNone(s._cache_element)
        self.assertFalse(s.is_read_only())

    def test_read_only_with_writable_cache(self):
        """ Test that read-only status reflects a non-read-only cache """
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement(readonly=False)
        self.assertFalse(s.is_read_only())

    def test_read_only_with_read_only_cache(self):
        """ Test that read-only status reflects a read-only cache """
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement(readonly=True)
        self.assertTrue(s.is_read_only())

    def test_has_invalid_key(self):
        """ Test that has-key is false for a key not in the table. """
        s = MemoryKeyValueStore()
        self.assertEqual(s._table, {})
        self.assertFalse(s.has('some key'))

    def test_has_key(self):
        """ Test that has-key returns true for entered keys. """
        s = MemoryKeyValueStore()
        s._table = {
            'a': 0,
            'b': 1,
            0: 2,
        }
        self.assertTrue(s.has('a'))
        self.assertTrue(s.has('b'))
        self.assertTrue(s.has(0))
        self.assertFalse(s.has('c'))

    def test_add_invalid_key(self):
        """ Test that we cannot add non-hashable keys. """
        s = MemoryKeyValueStore()

        self.assertRaises(TypeError, s.add, [1, 2, 3], 0)
        self.assertEqual(s._table, {})

        self.assertRaises(TypeError, s.add, {0: 1}, 0)
        self.assertEqual(s._table, {})

    def test_add_read_only(self):
        """
        Test that we cannot add when read-only (based on cache element).
        """
        s = MemoryKeyValueStore(DataMemoryElement(readonly=True))

        self.assertRaises(ReadOnlyError, s.add, 'a', 'b')
        self.assertRaises(ReadOnlyError, s.add, 'foo', None)

    def test_add(self):
        """ Test that we can add key-value pairs. """
        s = MemoryKeyValueStore()

        s.add('a', 'b')
        self.assertEqual(s._table, {'a': 'b'})

        s.add('foo', None)
        self.assertEqual(s._table, {
            'a': 'b',
            'foo': None,
        })

        s.add(0, 89)
        self.assertEqual(s._table, {
            'a': 'b',
            'foo': None,
            0: 89,
        })

    def test_add_with_caching(self):
        """
        Test that we can add key-value pairs and they reflect in the cache
        element.
        """
        c = DataMemoryElement()
        s = MemoryKeyValueStore(c)

        expected_cache_dict = {'a': 'b', 'foo': None, 0: 89}

        s.add('a', 'b')
        s.add('foo', None)
        s.add(0, 89)
        self.assertEqual(
            pickle.loads(c.get_bytes()),
            expected_cache_dict
        )

    def test_add_many_readonly(self):
        """ Test that we can't add many on a read-only instance. """
        s = MemoryKeyValueStore(DataMemoryElement(readonly=True))
        self.assertRaises(
            ReadOnlyError,
            s.add_many, {0: 1}
        )

    def test_add_many(self):
        """
        Test that we can add many key-values via a dictionary input.
        """
        d = {
            'a': 'b',
            'foo': None,
            0: 89,
        }

        s = MemoryKeyValueStore()
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, {})

        s.add_many(d)
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, d)

    def test_add_many_with_caching(self):
        """
        Test that adding many reflects in cache.
        """
        d = {
            'a': 'b',
            'foo': None,
            0: 89,
        }
        c = DataMemoryElement()

        s = MemoryKeyValueStore(c)
        self.assertEqual(s._table, {})
        self.assertEqual(c.get_bytes(), six.b(""))

        s.add_many(d)
        self.assertEqual(s._table, d)
        self.assertEqual(
            pickle.loads(c.get_bytes()),
            d
        )

    def test_remove_readonly(self):
        """ Test that we cannot remove from a read-only instance. """
        s = MemoryKeyValueStore(DataMemoryElement(readonly=True))
        self.assertRaises(
            ReadOnlyError,
            s.remove, 0
        )

    def test_remove_missing_key(self):
        """
        Test that we cannot remove a key not in the store.
        """
        s = MemoryKeyValueStore()
        s._table = {
            0: 1,
            'a': 'b'
        }
        self.assertRaises(
            KeyError,
            s.remove, 'some-key'
        )
        # table should remain unchanged.
        self.assertDictEqual(s._table, {0: 1, 'a': 'b'})

    def test_remove_with_cache(self):
        """
        Test that removal correctly updates the cache element.
        """
        existing_data = {
            0: 1,
            'a': 'b',
        }

        c = DataMemoryElement(pickle.dumps(existing_data))
        s = MemoryKeyValueStore(c)
        self.assertDictEqual(s._table, existing_data)

        s.remove('a')
        self.assertDictEqual(s._table, {0: 1})
        self.assertDictEqual(pickle.loads(c.get_bytes()),
                             {0: 1})

    def test_remove(self):
        """ Test normal removal. """
        s = MemoryKeyValueStore()
        s._table = {
            0: 1,
            'a': 'b',
        }

        s.remove(0)
        self.assertDictEqual(s._table, {'a': 'b'})

    def test_remove_many_readonly(self):
        """ Test that we cannot remove many from a read-only instance. """
        s = MemoryKeyValueStore(DataMemoryElement(readonly=True))
        self.assertRaises(
            ReadOnlyError,
            s.remove_many, [0]
        )

    def test_remove_many_missing_key(self):
        """
        Test that we cannot remove keys not present in table and that table
        is not modified on error.
        """
        expected_table = {
            0: 0,
            1: 1,
            2: 2,
        }

        s = MemoryKeyValueStore()
        s._table = {
            0: 0,
            1: 1,
            2: 2,
        }

        self.assertRaisesRegexp(
            KeyError, 'a',
            s.remove_many, ['a']
        )
        self.assertDictEqual(s._table, expected_table)

        # Even if one of the keys is value, the table should not be modified if
        # one of the keys is invalid.
        self.assertRaisesRegexp(
            KeyError, '6',
            s.remove_many, [1, 6]
        )
        self.assertDictEqual(s._table, expected_table)

        PY2_SET_KEY_ERROR_RE = "set\(\[(?:7|8), (?:7|8)\]\)"
        PY3_SET_KEY_ERROR_RE = "{(?:7|8), (?:7|8)}"
        self.assertRaisesRegexp(
            KeyError,
            # Should show a "set" that contains 7 and 8, regardless of order.
            '(?:{}|{})'.format(PY2_SET_KEY_ERROR_RE, PY3_SET_KEY_ERROR_RE),
            s.remove_many, [7, 8]
        )

    def test_remove_many(self):
        """
        Test expected remove_many functionality.
        """
        s = MemoryKeyValueStore()
        s._table = {
            0: 0,
            1: 1,
            2: 2,
        }

        s.remove_many([0, 1])
        self.assertDictEqual(s._table, {2: 2})

    def test_remove_many_with_cache(self):
        starting_table = {
            0: 0,
            1: 1,
            2: 2,
        }
        c = DataMemoryElement(pickle.dumps(starting_table))
        s = MemoryKeyValueStore(c)
        self.assertDictEqual(s._table, starting_table)

        s.remove_many([0, 2])

        self.assertDictEqual(pickle.loads(c.get_bytes()), {1: 1})

    def test_get_invalid_key(self):
        """
        Test that trying to get a value from a key not entered yields an
        exception.
        """
        s = MemoryKeyValueStore()
        self.assertRaises(
            KeyError,
            s.get, 0
        )

    def test_get_invalid_key_with_default(self):
        """ Test default value return on missing key. """
        s = MemoryKeyValueStore()
        self.assertEqual(
            s.get(0, 1),
            1,
        )
        assert s.get(0, ()) == ()

    def test_get_invalid_key_with_default_None(self):
        """ Test that passing None as a default returns None appropriately. """
        s = MemoryKeyValueStore()
        self.assertIsNone(s.get(0, None))

    def test_get(self):
        """ Test normal get functionality. """
        s = MemoryKeyValueStore()
        s._table['a'] = 'b'
        s._table[0] = 1

        assert s.get('a') == 'b'
        assert s.get(0) == 1

    def test_clear_readonly(self):
        """ Test trying to clear on a read-only store. """
        table_before_clear = dict(a=1, b=2, c=3)

        s = MemoryKeyValueStore()
        s._table = table_before_clear
        s.is_read_only = mock.MagicMock(return_value=True)

        self.assertRaises(
            ReadOnlyError,
            s.clear
        )
        self.assertEqual(s._table, table_before_clear)

    def test_clear(self):
        """ Test normal clear functionality. """
        table_before_clear = dict(a=1, b=2, c=3)

        s = MemoryKeyValueStore()
        s._table = table_before_clear
        s.clear()
        self.assertEqual(s._table, {})
