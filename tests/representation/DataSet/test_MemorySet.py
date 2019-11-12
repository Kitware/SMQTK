from __future__ import division, print_function

import pickle
import unittest

import six

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.data_element.memory_element import (
    BYTES_CONFIG_ENCODING,
    DataMemoryElement,
)
from smqtk.representation.data_set.memory_set import DataMemorySet


class TestDataFileSet (unittest.TestCase):

    def test_is_usable(self):
        # no dependencies
        self.assertTrue(DataMemorySet.is_usable())

    def test_default_config(self):
        default_config = DataMemorySet.get_default_config()
        self.assertEqual(len(default_config), 2)
        self.assertIn('cache_element', default_config)
        self.assertIsInstance(default_config['cache_element'],
                              dict)
        self.assertIsNone(default_config['cache_element']['type'])
        self.assertIn('pickle_protocol', default_config)

    def test_from_config_default(self):
        # From default configuration, which should be valid. Specifies no cache
        # pickle protocol -1.
        c = DataMemorySet.get_default_config()
        i = DataMemorySet.from_config(c)
        self.assertIsNone(i.cache_element)
        self.assertEqual(i.pickle_protocol, -1)
        self.assertEqual(i._element_map, {})

    def test_from_config_empty_cache(self):
        # Specify a memory element cache with no pre-existing bytes.
        c = DataMemorySet.get_default_config()
        c['cache_element']['type'] = 'DataMemoryElement'
        i = DataMemorySet.from_config(c)
        self.assertIsNotNone(i.cache_element)
        self.assertIsInstance(i.cache_element, DataMemoryElement)
        self.assertEqual(i.cache_element.get_bytes(), six.b(''))
        self.assertEqual(i.pickle_protocol, -1)
        self.assertEqual(i._element_map, {})

    def test_from_config_with_cache(self):
        # Use a cache element with bytes defining pickle of map to use.
        expected_map = dict(a=1, b=2, c=3)

        c = DataMemorySet.get_default_config()
        c['cache_element']['type'] = 'DataMemoryElement'
        c['cache_element']['DataMemoryElement']['bytes'] = \
            pickle.dumps(expected_map).decode(BYTES_CONFIG_ENCODING)

        i = DataMemorySet.from_config(c)

        self.assertIsInstance(i.cache_element, DataMemoryElement)
        self.assertEqual(i.pickle_protocol, -1)
        self.assertEqual(i._element_map, expected_map)

    def test_init_no_cache(self):
        i = DataMemorySet()
        self.assertIsNone(i.cache_element)
        self.assertEqual(i._element_map, {})
        self.assertEqual(i.pickle_protocol, -1)

    def test_init_empty_cache(self):
        cache_elem = DataMemoryElement()
        i = DataMemorySet(cache_elem, 2)
        self.assertEqual(i.cache_element, cache_elem)
        self.assertEqual(i.pickle_protocol, 2)
        self.assertEqual(i._element_map, {})

    def test_init_with_cache(self):
        expected_map = dict(a=1, b=2, c=3)
        expected_cache = DataMemoryElement(bytes=pickle.dumps(expected_map))

        i = DataMemorySet(expected_cache)

        self.assertEqual(i.cache_element, expected_cache)
        self.assertEqual(i.pickle_protocol, -1)
        self.assertEqual(i._element_map, expected_map)

    def test_iter(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }
        expected_map_values = {'a', 'b', 'c'}

        dms = DataMemorySet()
        dms._element_map = expected_map
        self.assertEqual(set(dms), expected_map_values)
        self.assertEqual(set(iter(dms)), expected_map_values)

    def test_caching_no_map_no_cache(self):
        dms = DataMemorySet()
        # should do nothing
        dms.cache()
        self.assertIsNone(dms.cache_element)
        self.assertEqual(dms._element_map, {})

    def test_cacheing_no_map(self):
        dms = DataMemorySet(DataMemoryElement())
        dms.cache()
        # technically caches something, but that something is an empty map.
        self.assertFalse(dms.cache_element.is_empty())
        self.assertEqual(pickle.loads(dms.cache_element.get_bytes()), {})

    def test_cacheing_with_map(self):
        expected_cache = DataMemoryElement()
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet(expected_cache)
        dms._element_map = expected_map
        dms.cache()

        self.assertFalse(expected_cache.is_empty())
        self.assertEqual(pickle.loads(expected_cache.get_bytes()), expected_map)

    def test_caching_readonly_cache(self):
        ro_cache = DataMemoryElement(readonly=True)
        dms = DataMemorySet(ro_cache)
        self.assertRaises(
            ReadOnlyError,
            dms.cache
        )

    def test_get_config_from_config_idempotence(self):
        default_c = DataMemorySet.get_default_config()
        self.assertEqual(
            DataMemorySet.from_config(default_c).get_config(),
            default_c
        )

        c = DataMemorySet.get_default_config()
        c['cache_element']['type'] = 'DataMemoryElement'
        c['cache_element']['DataMemoryElement']['readonly'] = True
        c['pickle_protocol'] = 1
        self.assertEqual(
            DataMemorySet.from_config(c).get_config(),
            c
        )

    def test_count(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet()
        dms._element_map = expected_map
        self.assertEqual(dms.count(), 3)

    def test_uuids(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet()
        dms._element_map = expected_map
        self.assertEqual(dms.uuids(), {0, 75, 124769})

    def test_has_uuid(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet()
        dms._element_map = expected_map
        self.assertTrue(dms.has_uuid(0))
        self.assertTrue(dms.has_uuid(75))
        self.assertTrue(dms.has_uuid(124769))

    def test_add_data_not_DataElement(self):
        dms = DataMemorySet()
        self.assertRaises(
            AssertionError,
            dms.add_data, "not data element"
        )

    def test_add_data(self):
        de = DataMemoryElement(six.b('some bytes'), 'text/plain', True)
        expected_map = {de.uuid(): de}

        dms = DataMemorySet()
        dms.add_data(de)
        self.assertEqual(dms._element_map, expected_map)

    def test_get_data_invalid_uuid(self):
        dms = DataMemorySet()
        self.assertRaises(
            KeyError,
            dms.get_data, 'invalid uuid'
        )

    def test_get_data_valid_uuid(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet()
        dms._element_map = expected_map
        self.assertEqual(dms.get_data(0), 'a')
        self.assertEqual(dms.get_data(75), 'b')
        self.assertEqual(dms.get_data(124769), 'c')
