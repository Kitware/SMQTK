import unittest

import six

from smqtk.representation.data_element.memory_element \
    import DataMemoryElement
from smqtk.representation.data_set.kvstore_backed \
    import KVSDataSet, DFLT_KVSTORE
from smqtk.representation.key_value.memory import MemoryKeyValueStore
from smqtk.utils.configuration import configuration_test_helper


class TestKeyValueDataSet (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.MEM_STORE = MemoryKeyValueStore()
        cls.MEM_STORE.add_many({
            0: 'a',
            1: 'b',
            'key': 'value'
        })

    def test_is_usable(self):
        # Should always be true
        self.assertTrue(KVSDataSet.is_usable())

    def test_get_default_config(self):
        # Default config should use a cache-less in-memory kvstore
        dflt = KVSDataSet.get_default_config()
        mkvs_key = "smqtk.representation.key_value.memory.MemoryKeyValueStore"
        self.assertIn('kvstore', dflt)
        self.assertEqual(dflt['kvstore']['type'], mkvs_key)
        # in-memory impl configuration should be the same as the default.
        self.assertEqual(
            dflt['kvstore'][mkvs_key],
            DFLT_KVSTORE.get_config()
        )

    def test_configuration(self):
        """ Test instance standard configuration """
        inst = KVSDataSet(self.MEM_STORE)
        for i in configuration_test_helper(inst):  # type: KVSDataSet
            assert isinstance(i._kvstore, MemoryKeyValueStore)
            assert i._kvstore.get_config() == self.MEM_STORE.get_config()

    def test_iter(self):
        kvds = KVSDataSet(self.MEM_STORE)
        self.assertEqual(set(kvds), set(self.MEM_STORE.keys()))

    def test_count(self):
        kvds = KVSDataSet(self.MEM_STORE)
        self.assertEqual(kvds.count(), 3)

    def test_uuids(self):
        kvds = KVSDataSet(self.MEM_STORE)
        self.assertEqual(
            kvds.uuids(),
            set(self.MEM_STORE.keys())
        )

    def test_has_uuid(self):
        kvds = KVSDataSet(self.MEM_STORE)
        self.assertTrue(kvds.has_uuid(0))
        self.assertTrue(kvds.has_uuid(1))
        self.assertTrue(kvds.has_uuid('key'))

    def test_has_uuid_invalid_key(self):
        kvds = KVSDataSet(self.MEM_STORE)
        self.assertFalse(kvds.has_uuid(4))
        self.assertFalse(kvds.has_uuid('NOT A KEY'))

    def test_add_data_not_dataelement(self):
        kvds = KVSDataSet()
        self.assertRaises(
            ValueError,
            kvds.add_data, 'one', 'two'
        )

    def test_add_data(self):
        mem_kv = MemoryKeyValueStore()
        kvds = KVSDataSet(mem_kv)

        de1 = DataMemoryElement(six.b('bytes1'))
        de2 = DataMemoryElement(six.b('bytes2'))
        kvds.add_data(de1, de2)

        # Check that appropriate keys and values are retrievable and located in
        # used KV-store.
        self.assertIn(de1.uuid(), mem_kv)
        self.assertIn(de2.uuid(), mem_kv)
        self.assertEqual(mem_kv.get(de1.uuid()), de1)
        self.assertEqual(mem_kv.get(de2.uuid()), de2)

    def test_get_data_bad_uuids(self):
        # get_data doesn't check that what is being returned ins a data
        # element, so we just use the cls.MEM_STORE for now.
        kvds = KVSDataSet(self.MEM_STORE)
        self.assertEqual(kvds.get_data(0), 'a')
        self.assertEqual(kvds.get_data(1), 'b')
        self.assertEqual(kvds.get_data('key'), 'value')
