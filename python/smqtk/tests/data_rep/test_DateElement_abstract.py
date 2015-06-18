"""
Tests for DataElement abstract interface class methods that provide
functionality.
"""
__author__ = 'purg'

import mock
import nose.tools as ntools
import unittest

import smqtk.data_rep.data_element_abstract


class DummyDataElement (smqtk.data_rep.data_element_abstract.DataElement):
    # abstract methods have no base functionality

    def content_type(self):
        return 'text/plain'

    def get_bytes(self):
        return "hello world"

    def uuid(self):
        return self.md5()


class TestDataElementAbstract (unittest.TestCase):

    def setUp(self):
        self.de = DummyDataElement()

        self.expected_md5 = '5eb63bbbe01eeed093cb22bb8f5acdc3'

    def test_md5(self):
        ntools.assert_is_none(self.de._md5_cache)

        md5 = self.de.md5()

        ntools.assert_is_not_none(self.de._md5_cache)
        ntools.assert_equal(self.de._md5_cache, self.expected_md5)
        ntools.assert_equal(md5, self.expected_md5)

        # When called a second time, should use cache instead of recomputing
        with mock.patch("smqtk.data_rep.data_element_abstract.hashlib") as mock_hashlib:
            md5 = self.de.md5()
            ntools.assert_false(mock_hashlib.md5.called)
            ntools.assert_equal(md5, self.expected_md5)

    @mock.patch('__builtin__.file')
    @mock.patch('smqtk.data_rep.data_element_abstract.safe_create_dir')
    @mock.patch('__builtin__.open')
    @mock.patch('smqtk.data_rep.data_element_abstract.tempfile')
    def test_writeTemp(self, mock_tempfile, mock_open, mock_scd,
                       mock_file):
        expected_tempfile_path = '/path/to/tmpfile'

        mock_tempfile.mkstemp.return_value = (0, expected_tempfile_path)
        mock_open.return_value = mock_file()

        with mock.patch('smqtk.data_rep.data_element_abstract.os'):
            path = self.de.write_temp()

        ntools.assert_false(mock_scd.called)
        ntools.assert_true(mock_tempfile.mkstemp.called)
        mock_open.assert_called_once_with(expected_tempfile_path, 'wb')
        mock_file().__enter__().write.assert_called_once_with(self.de.get_bytes())
        ntools.assert_equal(path, expected_tempfile_path)

        # Running it again should result in the same return, but system calls
        # should be made (caching)
        mock_tempfile.reset_mock()
        mock_open.reset_mock()
        mock_scd.reset_mock()
        mock_file.reset_mock()

        with mock.patch('smqtk.data_rep.data_element_abstract.os'):
            path = self.de.write_temp()

        ntools.assert_false(mock_scd.called)
        ntools.assert_false(mock_tempfile.mkstemp.called)
        ntools.assert_false(mock_open.called)
        ntools.assert_false(mock_file().__enter__().write.called)
        ntools.assert_equal(path, expected_tempfile_path)

    @mock.patch("smqtk.data_rep.data_element_abstract.os")
    def test_cleanTemp_noTemp(self, mock_os):
        # should do all of nothing
        ntools.assert_false(hasattr(self.de, '_temp_filepath'))

        self.de.clean_temp()

        ntools.assert_false(mock_os.path.isfile.called)
        ntools.assert_false(mock_os.remove.called)

    @mock.patch("smqtk.data_rep.data_element_abstract.os")
    def test_cleanTemp_hasTemp_noPath(self, mock_os):
        self.de._temp_filepath = None
        mock_os.path.isfile.return_value = False

        self.de.clean_temp()

        ntools.assert_true(mock_os.path.isfile.called)
        mock_os.path.isfile.assert_called_once_with(None)
        ntools.assert_false(mock_os.remove.called)

    @mock.patch("smqtk.data_rep.data_element_abstract.os")
    def test_cleanTemp_hasTemp_noPath(self, mock_os):
        expected_path = '/tmp/something'
        self.de._temp_filepath = expected_path
        mock_os.path.isfile.return_value = True

        self.de.clean_temp()

        ntools.assert_true(mock_os.path.isfile.called)
        mock_os.path.isfile.assert_called_once_with(expected_path)
        ntools.assert_true(mock_os.remove.called)
        mock_os.remove.assert_called_once_with(expected_path)
