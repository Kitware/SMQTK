import mock
import nose.tools as ntools
import unittest

from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_set.memory_set import DataMemorySet


class TestDataFileSet (unittest.TestCase):

    def test_configuration(self):
        default_config = DataMemorySet.get_default_config()
        expected_config = {
            "file_cache": None,
            "pickle_protocol": -1,
        }
        ntools.assert_equal(default_config, expected_config)

        inst1 = DataMemorySet.from_config(default_config)
        # idempotency
        ntools.assert_equal(default_config, inst1.get_config())

        inst2 = DataMemorySet.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)

    @mock.patch('smqtk.representation.data_set.memory_set.os.path.isfile')
    @mock.patch('smqtk.representation.data_set.memory_set.pickle')
    @mock.patch('__builtin__.open')
    def test_new_from_cache(self, m_open, m_pickle, m_isfile):
        expected_cache_path = '/some/file/path'
        expected_pickle_load_return = {'key': 'value'}

        # so os.path.isfile returns true for our mock path
        def expected_isfile_effect(path):
            if path == expected_cache_path:
                return True
            raise RuntimeError("Did not get expected path.")

        m_pickle.load.return_value = expected_pickle_load_return
        m_isfile.side_effect = expected_isfile_effect

        dms = DataMemorySet(expected_cache_path)
        m_open.assert_called_once_with(expected_cache_path)
        ntools.assert_equal(dms._element_map, expected_pickle_load_return)

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

    @mock.patch('smqtk.representation.data_set.memory_set.pickle')
    @mock.patch('smqtk.representation.data_set.memory_set.safe_file_write')
    def test_caching_with_filepath(self, m_sfw, m_pickle):
        expected_cache_path = '/some/file/path'
        expected_pickle_return = 'some pickle value'

        m_pickle.dumps.return_value = expected_pickle_return

        dms = DataMemorySet(expected_cache_path)
        dms.cache()

        m_sfw.assert_called_once_with(expected_cache_path,
                                      expected_pickle_return)

    @mock.patch('smqtk.representation.data_set.memory_set.pickle')
    @mock.patch('smqtk.representation.data_set.memory_set.safe_file_write')
    def test_caching_no_filepath(self, m_sfw, m_pickle):
        # should not do anything
        dms = DataMemorySet()
        dms.cache()
        ntools.assert_equal(m_sfw.call_count, 0)
        ntools.assert_equal(m_pickle.call_count, 0)

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
