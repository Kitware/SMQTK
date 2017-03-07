import nose.tools as ntools
import pickle
import unittest

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_set.memory_set import DataMemorySet


class TestDataFileSet (unittest.TestCase):

    def test_is_usable(self):
        # no dependencies
        ntools.assert_true(DataMemorySet.is_usable())

    def test_default_config(self):
        default_config = DataMemorySet.get_default_config()
        ntools.assert_equal(len(default_config), 2)
        ntools.assert_in('cache_element', default_config)
        ntools.assert_is_instance(default_config['cache_element'],
                                  dict)
        ntools.assert_is_none(default_config['cache_element']['type'])
        ntools.assert_in('pickle_protocol', default_config)

    def test_from_config_default(self):
        # From default configuration, which should be valid. Specifies no cache
        # pickle protocol -1.
        c = DataMemorySet.get_default_config()
        i = DataMemorySet.from_config(c)
        ntools.assert_is_none(i.cache_element)
        ntools.assert_equal(i.pickle_protocol, -1)
        ntools.assert_equal(i._element_map, {})

    def test_from_config_empty_cache(self):
        # Specify a memory element cache with no pre-existing bytes.
        c = DataMemorySet.get_default_config()
        c['cache_element']['type'] = 'DataMemoryElement'
        i = DataMemorySet.from_config(c)
        ntools.assert_is_not_none(i.cache_element)
        ntools.assert_is_instance(i.cache_element, DataMemoryElement)
        ntools.assert_equal(i.cache_element.get_bytes(), '')
        ntools.assert_equal(i.pickle_protocol, -1)
        ntools.assert_equal(i._element_map, {})

    def test_from_config_with_cache(self):
        # Use a cache element with bytes defining pickle of map to use.
        expected_map = dict(a=1, b=2, c=3)

        c = DataMemorySet.get_default_config()
        c['cache_element']['type'] = 'DataMemoryElement'
        c['cache_element']['DataMemoryElement']['bytes'] = \
            pickle.dumps(expected_map)

        i = DataMemorySet.from_config(c)

        ntools.assert_is_instance(i.cache_element, DataMemoryElement)
        ntools.assert_equal(i.pickle_protocol, -1)
        ntools.assert_equal(i._element_map, expected_map)

    def test_init_no_cache(self):
        i = DataMemorySet()
        ntools.assert_is_none(i.cache_element)
        ntools.assert_equal(i._element_map, {})
        ntools.assert_equal(i.pickle_protocol, -1)

    def test_init_empty_cache(self):
        cache_elem = DataMemoryElement()
        i = DataMemorySet(cache_elem, 2)
        ntools.assert_equal(i.cache_element, cache_elem)
        ntools.assert_equal(i.pickle_protocol, 2)
        ntools.assert_equal(i._element_map, {})

    def test_init_with_cache(self):
        expected_map = dict(a=1, b=2, c=3)
        expected_cache = DataMemoryElement(bytes=pickle.dumps(expected_map))

        i = DataMemorySet(expected_cache)

        ntools.assert_equal(i.cache_element, expected_cache)
        ntools.assert_equal(i.pickle_protocol, -1)
        ntools.assert_equal(i._element_map, expected_map)

    def test_iter(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }
        expected_map_values = {'a', 'b', 'c'}

        dms = DataMemorySet()
        dms._element_map = expected_map
        ntools.assert_equal(set(dms), expected_map_values)
        ntools.assert_equal(set(iter(dms)), expected_map_values)

    def test_caching_no_map_no_cache(self):
        dms = DataMemorySet()
        # should do nothing
        dms.cache()
        ntools.assert_is_none(dms.cache_element)
        ntools.assert_equal(dms._element_map, {})

    def test_cacheing_no_map(self):
        dms = DataMemorySet(DataMemoryElement())
        dms.cache()
        # technically caches something, but that something is an empty map.
        ntools.assert_false(dms.cache_element.is_empty())
        ntools.assert_equal(pickle.loads(dms.cache_element.get_bytes()),
                            {})

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

        ntools.assert_false(expected_cache.is_empty())
        ntools.assert_equal(pickle.loads(expected_cache.get_bytes()),
                            expected_map)

    def test_caching_readonly_cache(self):
        ro_cache = DataMemoryElement(readonly=True)
        dms = DataMemorySet(ro_cache)
        ntools.assert_raises(
            ReadOnlyError,
            dms.cache
        )

    def test_get_config_from_config_idempotence(self):
        default_c = DataMemorySet.get_default_config()
        ntools.assert_equal(
            DataMemorySet.from_config(default_c).get_config(),
            default_c
        )

        c = DataMemorySet.get_default_config()
        c['cache_element']['type'] = 'DataMemoryElement'
        c['cache_element']['DataMemoryElement']['readonly'] = True
        c['pickle_protocol'] = 1
        ntools.assert_equal(
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
        ntools.assert_equal(dms.count(), 3)

    def test_uuids(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet()
        dms._element_map = expected_map
        ntools.assert_equal(dms.uuids(), {0, 75, 124769})

    def test_has_uuid(self):
        expected_map = {
            0: 'a',
            75: 'b',
            124769: 'c',
        }

        dms = DataMemorySet()
        dms._element_map = expected_map
        ntools.assert_true(dms.has_uuid(0))
        ntools.assert_true(dms.has_uuid(75))
        ntools.assert_true(dms.has_uuid(124769))

    def test_add_data_not_DataElement(self):
        dms = DataMemorySet()
        ntools.assert_raises(
            AssertionError,
            dms.add_data, "not data element"
        )

    def test_add_data(self):
        de = DataMemoryElement('some bytes', 'text/plain', True)
        expected_map = {de.uuid(): de}

        dms = DataMemorySet()
        dms.add_data(de)
        ntools.assert_equal(dms._element_map, expected_map)

    def test_get_data_invalid_uuid(self):
        dms = DataMemorySet()
        ntools.assert_raises(
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
        ntools.assert_equal(dms.get_data(0), 'a')
        ntools.assert_equal(dms.get_data(75), 'b')
        ntools.assert_equal(dms.get_data(124769), 'c')
