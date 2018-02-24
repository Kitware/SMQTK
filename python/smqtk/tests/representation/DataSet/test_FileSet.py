from __future__ import division, print_function
import mock
import nose.tools
import os
import six
import unittest

from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_set.file_set import DataFileSet
from smqtk.tests import TEST_DATA_DIR


class TestDataFileSet (unittest.TestCase):

    def test_configuration(self):
        default_config = DataFileSet.get_default_config()
        nose.tools.assert_equal(default_config, {
            'root_directory': None,
            'uuid_chunk': 10,
            'pickle_protocol': -1,
        })

        default_config['root_directory'] = '/some/dir'
        inst1 = DataFileSet.from_config(default_config)
        nose.tools.assert_equal(default_config, inst1.get_config())

        inst2 = DataFileSet.from_config(inst1.get_config())
        nose.tools.assert_equal(inst1, inst2)

    def test_new_invalid_uuid_chunk(self):
        nose.tools.assert_raises(
            ValueError,
            DataFileSet, '/', 0
        )

        nose.tools.assert_raises(
            ValueError,
            DataFileSet, '/', -1
        )

    def test_new(self):
        # The following should be valid constructor parameter setups
        DataFileSet('/')
        DataFileSet('/', uuid_chunk=None)
        DataFileSet('/', uuid_chunk=1)
        DataFileSet('/', uuid_chunk=2346)

        DataFileSet('relative/path')
        DataFileSet('relative/path', uuid_chunk=None)
        DataFileSet('relative/path', uuid_chunk=1)
        DataFileSet('relative/path', uuid_chunk=2346)

    def test_iter_file_tree_chunk0(self):
        test_dir_path = os.path.join(TEST_DATA_DIR, 'test_data_file_set_tree')
        expected_filepaths = {
            os.path.join(test_dir_path, 'UUID_0.dataElement'),
            os.path.join(test_dir_path, 'UUID_1.dataElement'),
            os.path.join(test_dir_path, 'UUID_2.dataElement'),
        }

        dfs = DataFileSet(test_dir_path, uuid_chunk=None, pickle_protocol=2)
        actual_filepaths = set(dfs._iter_file_tree())
        nose.tools.assert_set_equal(actual_filepaths, expected_filepaths)

    def test_iter_file_tree_chunk3(self):
        test_dir_path = os.path.join(TEST_DATA_DIR, 'test_data_file_set_tree')
        expected_filepaths = {
            os.path.join(test_dir_path, '0/0/UUID_000.dataElement'),
            os.path.join(test_dir_path, '0/0/UUID_001.dataElement'),
            os.path.join(test_dir_path, '0/1/UUID_012.dataElement'),
            os.path.join(test_dir_path, '1/8/UUID_180.dataElement'),
            os.path.join(test_dir_path, '3/1/UUID_317.dataElement'),
        }

        dfs = DataFileSet(test_dir_path, uuid_chunk=3, pickle_protocol=2)
        actual_filepaths = set(dfs._iter_file_tree())
        nose.tools.assert_set_equal(actual_filepaths, expected_filepaths)

    def test_iter_file_tree_chunk4(self):
        test_dir_path = os.path.join(TEST_DATA_DIR, 'test_data_file_set_tree')
        expected_filepaths = {
            os.path.join(test_dir_path, '4/3/2/1/UUID_43210.dataElement'),
        }

        dfs = DataFileSet(test_dir_path, uuid_chunk=5, pickle_protocol=2)
        actual_filepaths = set(dfs._iter_file_tree())
        nose.tools.assert_set_equal(actual_filepaths, expected_filepaths)

    def test_containing_dir_str_uuid(self):
        # Chunk == None
        s = DataFileSet('/', uuid_chunk=None)
        nose.tools.assert_equal(s._containing_dir('0000'), '/')
        nose.tools.assert_equal(s._containing_dir('346'), '/')
        # Chunk == 1
        s = DataFileSet('/', uuid_chunk=1)
        nose.tools.assert_equal(s._containing_dir('0000'), '/')
        nose.tools.assert_equal(s._containing_dir('346'), '/')
        # Chunk == 3
        s = DataFileSet('/', uuid_chunk=3)
        nose.tools.assert_equal(s._containing_dir('123456'), '/12/34')
        nose.tools.assert_equal(s._containing_dir('685225624578'), '/6852/2562')
        nose.tools.assert_equal(s._containing_dir('1234567'), '/123/45')

    def test_containing_dir_not_str_uuid(self):
        nose.tools.assert_equal(
            DataFileSet('/', None)._containing_dir(4123458),
            "/"
        )
        nose.tools.assert_equal(
            DataFileSet('/', 3)._containing_dir(4123458),
            "/412/34"
        )

    def test_fp_for_uuid(self):
        nose.tools.assert_equal(
            DataFileSet('/', None)._fp_for_uuid(0),
            '/UUID_0.dataElement'
        )
        nose.tools.assert_equal(
            DataFileSet('/', None)._fp_for_uuid('abc'),
            '/UUID_abc.dataElement'
        )
        nose.tools.assert_equal(
            DataFileSet('/', 3)._fp_for_uuid('abc'),
            '/a/b/UUID_abc.dataElement'
        )

    @mock.patch('smqtk.representation.data_set.file_set.pickle')
    @mock.patch('smqtk.representation.data_set.file_set.open', new_callable=mock.MagicMock)
    def test_iter(self, m_open, m_pickle):
        expected_file_tree_iter = ['/a', '/b', '/d']
        dfs = DataFileSet('/')
        dfs._iter_file_tree = mock.MagicMock(
            return_value=expected_file_tree_iter)
        list(dfs)
        nose.tools.assert_equal(m_open.call_count, 3)
        nose.tools.assert_equal(m_pickle.load.call_count, 3)
        m_open.assert_any_call('/a')
        m_open.assert_any_call('/b')
        m_open.assert_any_call('/d')

    def test_count(self):
        expected_file_tree_iter = ['/a', '/b', '/d']
        dfs = DataFileSet('/')
        dfs._iter_file_tree = mock.MagicMock(
            return_value=expected_file_tree_iter)
        nose.tools.assert_equal(dfs.count(), 3)

    def test_uuids(self):
        # mocking self iteration results
        expected_data_elements = [
            DataMemoryElement(six.b('a')),
            DataMemoryElement(six.b('b')),
            DataMemoryElement(six.b('v')),
        ]
        expected_uuid_set = {
            DataMemoryElement(six.b('a')).uuid(),
            DataMemoryElement(six.b('b')).uuid(),
            DataMemoryElement(six.b('v')).uuid(),
        }

        # Replacement iterator for DataFileSet to yield expected test values.
        def test_iter():
            for e in expected_data_elements:
                yield e

        with mock.patch('smqtk.representation.data_set.file_set.DataFileSet'
                        '.__iter__') as m_iter:
            m_iter.side_effect = test_iter

            dfs = DataFileSet('/')
            nose.tools.assert_set_equal(dfs.uuids(), expected_uuid_set)

    def test_add_data_not_dataelement(self):
        dfs = DataFileSet('/')
        nose.tools.assert_raises_regexp(
            AssertionError,
            "^Not given a DataElement for addition:",
            dfs.add_data, 'not a dataElement'
        )

    @mock.patch('smqtk.representation.data_set.file_set.pickle')
    @mock.patch('smqtk.representation.data_set.file_set.open')
    @mock.patch('smqtk.representation.data_set.file_set.file_utils'
                '.safe_create_dir')
    @mock.patch('smqtk.representation.data_set.file_set.isinstance')
    def test_add_data_single(self, m_isinstance, m_scd, m_open, m_pickle):
        # Pretend that we are giving DataElement instances
        m_isinstance.return_value = True

        # Testing that appropriate directories are given to safe_create_dir and
        # appropriate filepaths are passed to open.
        expected_uuid = 'abcd'

        mock_elem = mock.MagicMock()
        mock_elem.uuid.return_value = expected_uuid

        dfs = DataFileSet('/', uuid_chunk=None)
        dfs.add_data(mock_elem)
        m_scd.assert_called_with('/')
        m_open.assert_called_with('/UUID_abcd.dataElement', 'wb')

        dfs = DataFileSet('/', uuid_chunk=1)
        dfs.add_data(mock_elem)
        m_scd.assert_called_with('/')
        m_open.assert_called_with('/UUID_abcd.dataElement', 'wb')

        dfs = DataFileSet('/', uuid_chunk=2)
        dfs.add_data(mock_elem)
        m_scd.assert_called_with('/ab')
        m_open.assert_called_with('/ab/UUID_abcd.dataElement', 'wb')

    @mock.patch('smqtk.representation.data_set.file_set.pickle')
    @mock.patch('smqtk.representation.data_set.file_set.open')
    @mock.patch('smqtk.representation.data_set.file_set.file_utils'
                '.safe_create_dir')
    @mock.patch('smqtk.representation.data_set.file_set.isinstance')
    def test_add_data_multiple_chunk0(self, m_isinstance, m_scd, m_open,
                                      m_pickle):
        # Pretend that we are giving DataElement instances
        m_isinstance.return_value = True

        # Testing that appropriate directories are given to safe_create_dir and
        # appropriate filepaths are passed to open.
        expected_uuid_1 = "abcdefg"
        expected_uuid_2 = "1234567"
        expected_uuid_3 = "4F*s93#5"

        mock_elem_1 = mock.MagicMock()
        mock_elem_1.uuid.return_value = expected_uuid_1
        mock_elem_2 = mock.MagicMock()
        mock_elem_2.uuid.return_value = expected_uuid_2
        mock_elem_3 = mock.MagicMock()
        mock_elem_3.uuid.return_value = expected_uuid_3

        # Chunk = None
        dfs = DataFileSet('/', uuid_chunk=None)
        dfs.add_data(mock_elem_1, mock_elem_2, mock_elem_3)
        # Created root 3 times
        nose.tools.assert_equal(m_scd.call_count, 3)
        m_scd.assert_called_with('/')
        # called open correctly 3 times
        nose.tools.assert_equal(m_open.call_count, 3)
        m_open.assert_any_call('/UUID_abcdefg.dataElement', 'wb')
        m_open.assert_any_call('/UUID_1234567.dataElement', 'wb')
        m_open.assert_any_call('/UUID_4F*s93#5.dataElement', 'wb')

    @mock.patch('smqtk.representation.data_set.file_set.pickle')
    @mock.patch('smqtk.representation.data_set.file_set.open')
    @mock.patch('smqtk.representation.data_set.file_set.file_utils'
                '.safe_create_dir')
    @mock.patch('smqtk.representation.data_set.file_set.isinstance')
    def test_add_data_multiple_chunk3(self, m_isinstance, m_scd, m_open,
                                      m_pickle):
        # Pretend that we are giving DataElement instances
        m_isinstance.return_value = True

        # Testing that appropriate directories are given to safe_create_dir and
        # appropriate filepaths are passed to open.
        expected_uuid_1 = "abcdefg"
        expected_uuid_2 = "1234567"
        expected_uuid_3 = "4F*s93#5"

        mock_elem_1 = mock.MagicMock()
        mock_elem_1.uuid.return_value = expected_uuid_1
        mock_elem_2 = mock.MagicMock()
        mock_elem_2.uuid.return_value = expected_uuid_2
        mock_elem_3 = mock.MagicMock()
        mock_elem_3.uuid.return_value = expected_uuid_3

        # Chunk = 3
        dfs = DataFileSet('/', uuid_chunk=3)
        dfs.add_data(mock_elem_1, mock_elem_2, mock_elem_3)
        # Created correct directories
        nose.tools.assert_equal(m_scd.call_count, 3)
        m_scd.assert_any_call('/abc/de')
        m_scd.assert_any_call('/123/45')
        m_scd.assert_any_call('/4F*/s93')
        # called open correctly 3 times
        nose.tools.assert_equal(m_open.call_count, 3)
        m_open.assert_any_call('/abc/de/UUID_abcdefg.dataElement', 'wb')
        m_open.assert_any_call('/123/45/UUID_1234567.dataElement', 'wb')
        m_open.assert_any_call('/4F*/s93/UUID_4F*s93#5.dataElement', 'wb')

    @mock.patch('smqtk.representation.data_set.file_set.pickle')
    @mock.patch('smqtk.representation.data_set.file_set.open')
    @mock.patch('smqtk.representation.data_set.file_set.file_utils'
                '.safe_create_dir')
    @mock.patch('smqtk.representation.data_set.file_set.isinstance')
    def test_add_data_multiple_chunk3_relative(self, m_isinstance, m_scd, m_open,
                                      m_pickle):
        # Pretend that we are giving DataElement instances
        m_isinstance.return_value = True

        # Testing that appropriate directories are given to safe_create_dir and
        # appropriate filepaths are passed to open.
        expected_uuid_1 = "abcdefg"
        expected_uuid_2 = "1234567"
        expected_uuid_3 = "4F*s93#5"

        mock_elem_1 = mock.MagicMock()
        mock_elem_1.uuid.return_value = expected_uuid_1
        mock_elem_2 = mock.MagicMock()
        mock_elem_2.uuid.return_value = expected_uuid_2
        mock_elem_3 = mock.MagicMock()
        mock_elem_3.uuid.return_value = expected_uuid_3

        # Chunk = 3
        dfs = DataFileSet('rel/subdir', uuid_chunk=3)
        dfs.add_data(mock_elem_1, mock_elem_2, mock_elem_3)
        # Created correct directories
        nose.tools.assert_equal(m_scd.call_count, 3)
        m_scd.assert_any_call('rel/subdir/abc/de')
        m_scd.assert_any_call('rel/subdir/123/45')
        m_scd.assert_any_call('rel/subdir/4F*/s93')
        # called open correctly 3 times
        nose.tools.assert_equal(m_open.call_count, 3)
        m_open.assert_any_call('rel/subdir/abc/de/UUID_abcdefg.dataElement', 'wb')
        m_open.assert_any_call('rel/subdir/123/45/UUID_1234567.dataElement', 'wb')
        m_open.assert_any_call('rel/subdir/4F*/s93/UUID_4F*s93#5.dataElement', 'wb')

    @mock.patch('smqtk.representation.data_set.file_set.osp.isfile')
    def test_get_data_no_file(self, m_isfile):
        # Testing when we generate a filepath that does not point to an existing
        # file, meaning the UUID is referring to a dataElement not a part of our
        # set.
        m_isfile.return_value = False

        dfs = DataFileSet(TEST_DATA_DIR, None)
        nose.tools.assert_raises_regexp(
            KeyError,
            'no_exist_uuid',
            dfs.get_data, 'no_exist_uuid'
        )

    @mock.patch('smqtk.representation.data_set.file_set.pickle')
    @mock.patch('smqtk.representation.data_set.file_set.open')
    @mock.patch('smqtk.representation.data_set.file_set.osp.isfile')
    def test_get_data_valid_filepath(self, m_isfile, m_open, m_pickle):
        # Testing that filepath we get back from _fp_for_uuid generator is
        # valid, meaning that the given UUID does refer to a serialized
        # DataElement in our set, which is then opened and returned.
        m_isfile.return_value = True

        expected_uuid = 'abc'
        expected_filepath = os.path.join(TEST_DATA_DIR, 'UUID_abc.dataElement')
        expected_pickle_return = 'loaded DataElement instance'

        m_pickle.load.return_value = expected_pickle_return

        dfs = DataFileSet(TEST_DATA_DIR, None)
        actual_return = dfs.get_data(expected_uuid)
        m_isfile.assert_called_once_with(expected_filepath)
        m_open.assert_called_once_with(expected_filepath, 'rb')
        nose.tools.assert_equal(actual_return, expected_pickle_return)
