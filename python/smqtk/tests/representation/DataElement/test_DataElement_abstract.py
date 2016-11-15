"""
Tests for DataElement abstract interface class methods that provide
functionality.
"""
import hashlib
import mock
import nose.tools as ntools
import os.path as osp
import tempfile
import unittest

import smqtk.representation.data_element


__author__ = "paul.tunison@kitware.com"


# because this has a stable mimetype conversion
EXPECTED_CONTENT_TYPE = "image/png"
EXPECTED_BYTES = "hello world"
EXPECTED_MD5 = hashlib.md5(EXPECTED_BYTES).hexdigest()
EXPECTED_SHA1 = hashlib.sha1(EXPECTED_BYTES).hexdigest()
# UUID is currently set to be equivalent to SHA1 value by default
EXPECTED_UUID = EXPECTED_SHA1


# Caches the temp directory before we start mocking things out that would
# otherwise be required for the tempfile module to determine the temp directory.
tempfile.gettempdir()


class DummyDataElement (smqtk.representation.data_element.DataElement):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def content_type(self):
        return EXPECTED_CONTENT_TYPE

    def get_bytes(self):
        # Aligned with the checksum strings in test class setUp method
        return EXPECTED_BYTES


class TestDataElementAbstract (unittest.TestCase):

    def test_md5(self):
        de = DummyDataElement()
        ntools.assert_is_none(de._md5_cache)
        md5 = de.md5()

        ntools.assert_is_not_none(de._md5_cache)
        ntools.assert_equal(de._md5_cache, EXPECTED_MD5)
        ntools.assert_equal(md5, EXPECTED_MD5)

        # When called a second time, should use cache instead of recomputing
        with mock.patch("smqtk.representation.data_element.hashlib") as mock_hashlib:
            md5 = de.md5()
            ntools.assert_false(mock_hashlib.md5.called)
            ntools.assert_equal(md5, EXPECTED_MD5)

    def test_sha1(self):
        de = DummyDataElement()
        ntools.assert_is_none(de._sha1_cache)
        sha1 = de.sha1()

        ntools.assert_is_not_none(de._sha1_cache)
        ntools.assert_equal(de._sha1_cache, EXPECTED_SHA1)
        ntools.assert_equal(sha1, EXPECTED_SHA1)

        # When called a second time, should use cache instead of recomputing
        with mock.patch("smqtk.representation.data_element.hashlib") as mock_hashlib:
            sha1 = de.sha1()
            ntools.assert_false(mock_hashlib.sha1.called)
            ntools.assert_equal(sha1, EXPECTED_SHA1)

    def test_del(self):
        de = DummyDataElement()
        m_clean_temp = de.clean_temp = mock.Mock()
        del de

        ntools.assert_true(m_clean_temp.called)

    def test_uuid(self):
        de = DummyDataElement()
        ntools.assert_equal(de.uuid(), EXPECTED_UUID)

    def test_hashing(self):
        # Hash should be that of the UUID of the element
        de = DummyDataElement()
        ntools.assert_equal(hash(de), hash(EXPECTED_UUID))

    # Cases:
    #   - no existing temps, no specific dir
    #   - no existing temps, given specific dir
    #   - existing temps, no specific dir
    #   - existing temps, given specific dir
    #
    # Mocking open, os.open, os.close and fcntl to actual file interaction
    #   - os.open is used under the hood of tempfile to open a file (which also
    #       creates it on disk).

    @mock.patch('smqtk.representation.data_element.file_utils.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch('__builtin__.open')
    def test_writeTemp_noExisting_noDir(self,
                                        mock_open, mock_os_open, mock_os_close,
                                        mock_fcntl, mock_scd):
        # no existing temps, no specific dir
        fp = DummyDataElement().write_temp()

        ntools.assert_false(mock_scd.called)
        ntools.assert_true(mock_open.called)
        ntools.assert_equal(osp.dirname(fp), tempfile.gettempdir())

    @mock.patch('smqtk.representation.data_element.file_utils.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch('__builtin__.open')
    def test_writeTemp_noExisting_givenDir(self,
                                           mock_open, mock_os_open,
                                           mock_os_close, mock_fcntl, mock_scd):
        # no existing temps, given specific dir
        target_dir = '/some/dir/somewhere'

        fp = DummyDataElement().write_temp(target_dir)

        mock_scd.assert_called_once_with(target_dir)
        ntools.assert_true(mock_open.called)
        ntools.assert_not_equal(osp.dirname(fp), tempfile.gettempdir())
        ntools.assert_equal(osp.dirname(fp), target_dir)

    @mock.patch("smqtk.representation.data_element.file_element.osp.isfile")
    @mock.patch('smqtk.representation.data_element.file_utils.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch('__builtin__.open')
    def test_writeTemp_hasExisting_noDir(self,
                                         mock_open, mock_os_open, mock_os_close,
                                         mock_fcntl, mock_scd, mock_isfile):
        # Pretend we have existing temps. Will to "write" a temp file to no
        # specific dir, which should not write anything new and just return the
        # last path in the list.
        prev_0 = '/tmp/file.txt'
        prev_1 = '/tmp/file_two.png'

        de = DummyDataElement()
        de._temp_filepath_stack.append(prev_0)
        de._temp_filepath_stack.append(prev_1)

        # Make sure os.path.isfile returns true so we think things in temp stack
        # exist.
        simulate = True
        def osp_isfile_se(path):
            if simulate and path in {prev_0, prev_1}:
                return True
            else:
                return False
        mock_isfile.side_effect = osp_isfile_se

        fp = de.write_temp()

        ntools.assert_false(mock_scd.called)
        ntools.assert_false(mock_open.called)
        ntools.assert_equal(fp, prev_1)

        # _temp_filepath_stack files don't exist, so make sure isfile returns
        # false so clean_temp doesn't try to remove files that don't exist.
        simulate = False

    @mock.patch('smqtk.representation.data_element.file_utils.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch('__builtin__.open')
    def test_writeTemp_hasExisting_givenNewDir(self, mock_open, mock_os_open,
                                               mock_os_close, mock_fcntl,
                                               mock_scd):
        # existing temps, given specific dir
        prev_0 = '/tmp/file.txt'
        prev_1 = '/tmp/file_two.png'

        target_dir = '/some/specific/dir'

        de = DummyDataElement()
        de._temp_filepath_stack.append(prev_0)
        de._temp_filepath_stack.append(prev_1)

        fp = de.write_temp(temp_dir=target_dir)

        ntools.assert_true(mock_scd.called)
        ntools.assert_true(mock_open.called)
        ntools.assert_equal(osp.dirname(fp), target_dir)

    @mock.patch("smqtk.representation.data_element.file_element.osp.isfile")
    @mock.patch('smqtk.representation.data_element.file_utils.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch('__builtin__.open')
    def test_writeTemp_hasExisting_givenExistingDir(self, mock_open,
                                                    mock_os_open, mock_os_close,
                                                    mock_fcntl, mock_scd,
                                                    mock_isfile):
        # Pretend these files already exist as written temp files.
        # We test that write_temp with a target directory yields a previously
        #   "written" temp file.
        #
        # that given specific dir already in stack
        prev_0 = '/dir1/file.txt'
        prev_1 = '/tmp/things/file_two.png'
        prev_2 = '/some/specific/dir'

        simulate = True
        def osp_isfile_se(path):
            if simulate and path in {prev_0, prev_1, prev_2}:
                return True
            else:
                return False
        mock_isfile.side_effect = osp_isfile_se

        de = DummyDataElement()
        de._temp_filepath_stack.append(prev_0)
        de._temp_filepath_stack.append(prev_1)
        de._temp_filepath_stack.append(prev_2)

        target_dir = "/tmp/things"

        # Make sure os.path.isfile returns true so we think things in temp stack
        # exist.
        mock_isfile.return_value = True

        fp = de.write_temp(temp_dir=target_dir)

        ntools.assert_false(mock_scd.called)
        ntools.assert_false(mock_open.called)
        ntools.assert_equal(fp, prev_1)

        # _temp_filepath_stack files don't exist, so make sure isfile returns
        # false so clean_temp doesn't try to remove files that don't exist.
        simulate = False

    @mock.patch("smqtk.representation.data_element.os")
    def test_cleanTemp_noTemp(self, mock_os):
        # should do all of nothing
        de = DummyDataElement()

        de.clean_temp()

        ntools.assert_false(mock_os.path.isfile.called)
        ntools.assert_false(mock_os.remove.called)

    @mock.patch("smqtk.representation.data_element.os")
    def test_cleanTemp_hasTemp_badPath(self, mock_os):
        de = DummyDataElement()
        de._temp_filepath_stack.append('tmp/thing')
        mock_os.path.isfile.return_value = False

        de.clean_temp()

        mock_os.path.isfile.assert_called_once_with('tmp/thing')
        ntools.assert_false(mock_os.remove.called)

    @mock.patch("smqtk.representation.data_element.os")
    def test_cleanTemp_hasTemp_validPath(self, mock_os):
        expected_path = '/tmp/something'

        de = DummyDataElement()
        de._temp_filepath_stack.append(expected_path)
        mock_os.path.isfile.return_value = True

        de.clean_temp()

        mock_os.path.isfile.assert_called_once_with(expected_path)
        mock_os.remove.assert_called_once_with(expected_path)
