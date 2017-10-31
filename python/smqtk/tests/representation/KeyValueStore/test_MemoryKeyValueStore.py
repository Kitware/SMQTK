import pickle
import unittest

import mock

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.key_value.memory import MemoryKeyValueStore


class TestMemoryKeyValueStore (unittest.TestCase):

    def test_is_usable(self):
        # Should always be usable
        self.assertTrue(MemoryKeyValueStore.is_usable())

    def test_get_default(self):
        # Check default config
        default_config = MemoryKeyValueStore.get_default_config()
        self.assertIsInstance(default_config, dict)
        # - Should just contain cache element property, which is a nested plugin
        #   config with no default type.
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
        empty_dict_pickle = "(dp1\nS'some_key'\np2\nS'some_value'\np3\ns."

        # Test construction with memory data element.
        config = {'cache_element': {
            'DataMemoryElement': {
                'bytes': empty_dict_pickle,
            },
            'type': 'DataMemoryElement'
        }}
        s = MemoryKeyValueStore.from_config(config)
        self.assertIsInstance(s._cache_element, DataMemoryElement)
        self.assertEqual(s._table, expected_table)

    def test_new_no_cache(self):
        s = MemoryKeyValueStore()
        self.assertIsNone(s._cache_element)
        self.assertEqual(s._table, {})

    def test_new_empty_cache(self):
        # Cache element with no bytes.
        c = DataMemoryElement()
        s = MemoryKeyValueStore(c)
        self.assertEqual(s._cache_element, c)
        self.assertEqual(s._table, {})

    def test_new_cached_table(self):
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
        expected_repr = '<MemoryKeyValueStore cache_element: None>'
        s = MemoryKeyValueStore()
        actual_repr = repr(s)
        self.assertEqual(actual_repr, expected_repr)

    def test_repr_simple_cache(self):
        c = DataMemoryElement()
        s = MemoryKeyValueStore(c)
        expected_repr = "<MemoryKeyValueStore cache_element: " \
                        "DataMemoryElement{len(bytes): 0, content_type: " \
                        "None, readonly: False}>"
        self.assertEqual(repr(s), expected_repr)

    def test_count(self):
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
        s = MemoryKeyValueStore()
        s._cache_element = None
        # We expect an default DataElement config (no impl type defined)
        c = s.get_config()
        self.assertIn('cache_element', c)
        self.assertIsNone(c['cache_element']['type'])

    def test_get_config_mem_cache_elem(self):
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement('someBytes', 'text/plain', False)
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
        s = MemoryKeyValueStore()
        self.assertEqual(list(s.keys()), [])

    def test_keys_with_table(self):
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
        s = MemoryKeyValueStore()
        self.assertIsNone(s._cache_element)
        self.assertFalse(s.is_read_only())

    def test_read_only_with_writable_cache(self):
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement(readonly=False)
        self.assertFalse(s.is_read_only())

    def test_read_only_with_read_only_cache(self):
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement(readonly=True)
        self.assertTrue(s.is_read_only())

    def test_has_invalid_key(self):
        s = MemoryKeyValueStore()
        self.assertEqual(s._table, {})
        self.assertFalse(s.has('some key'))

    def test_has_valid_key(self):
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
        s = MemoryKeyValueStore()

        self.assertRaises(ValueError, s.add, [1, 2, 3], 0)
        self.assertEqual(s._table, {})

        self.assertRaises(ValueError, s.add, {0: 1}, 0)
        self.assertEqual(s._table, {})

    def test_add_read_only(self):
        s = MemoryKeyValueStore()
        s._cache_element = DataMemoryElement(readonly=True)

        self.assertRaises(ReadOnlyError, s.add, 'a', 'b')
        self.assertRaises(ReadOnlyError, s.add, 'foo', None)

    def test_add(self):
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

    def test_add_with_caching_no_cache(self):
        c = DataMemoryElement()
        s = MemoryKeyValueStore(c)

        expected_cache_dict = {'a': 'b', 'foo': None}

        s.add('a', 'b', False)
        # No caching means there should be nothign there yet
        self.assertEqual(
            c.get_bytes(),
            ""
        )

        s.add('foo', None, True)
        # With caching, meaning the state should be cached here, which includes
        # everything added previously, including the a:b pair.
        self.assertEqual(
            pickle.loads(c.get_bytes()),
            expected_cache_dict
        )

        s.add(0, 89, False)
        self.assertEqual(
            pickle.loads(c.get_bytes()),
            expected_cache_dict
        )

    def test_add_many(self):
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
        d = {
            'a': 'b',
            'foo': None,
            0: 89,
        }
        c = DataMemoryElement()

        s = MemoryKeyValueStore(c)
        self.assertEqual(s._table, {})
        self.assertEqual(c.get_bytes(), "")

        s.add_many(d)
        self.assertEqual(s._table, d)
        self.assertEqual(
            pickle.loads(c.get_bytes()),
            d
        )

    def test_get_invalid_key(self):
        s = MemoryKeyValueStore()
        self.assertRaises(
            KeyError,
            s.get, 0
        )

    def test_get_invalid_key_with_default(self):
        s = MemoryKeyValueStore()
        self.assertEqual(
            s.get(0, 1),
            1,
        )
        assert s.get(0, ()) == ()

    def test_get_invalid_key_with_default_None(self):
        s = MemoryKeyValueStore()
        self.assertIsNone(s.get(0, None))

    def test_get(self):
        s = MemoryKeyValueStore()
        s._table['a'] = 'b'
        s._table[0] = 1

        assert s.get('a') == 'b'
        assert s.get(0) == 1

    def test_clear(self):
        table_before_clear = dict(a=1, b=2, c=3)

        s = MemoryKeyValueStore()
        s._table = table_before_clear
        s.clear()
        self.assertEqual(s._table, {})

    def test_clear_readonly(self):
        table_before_clear = dict(a=1, b=2, c=3)

        s = MemoryKeyValueStore()
        s._table = table_before_clear
        s.is_read_only = mock.MagicMock(return_value=True)

        self.assertRaises(
            ReadOnlyError,
            s.clear
        )
        self.assertEqual(s._table, table_before_clear)
