import errno
import mock
import nose.tools as ntools
import os
import unittest

from smqtk.utils import file_utils


class TestSafeCreateDir (unittest.TestCase):

    @mock.patch('smqtk.utils.file_utils.os.makedirs')
    def test_noExists(self, mock_os_makedirs):
        dir_path = "/some/directory/somewhere"
        p = file_utils.safe_create_dir(dir_path)

        ntools.assert_true(mock_os_makedirs.called)
        ntools.assert_equals(p, dir_path)

    @mock.patch('smqtk.utils.file_utils.os.path.exists')
    @mock.patch('smqtk.utils.file_utils.os.makedirs')
    def test_existError_alreadyExists(self, mock_os_makedirs, mock_osp_exists):
        mock_os_makedirs.side_effect = OSError(errno.EEXIST,
                                               "Existing directory")

        mock_osp_exists.return_value = True

        dir_path = '/existing/dir'
        p = file_utils.safe_create_dir(dir_path)

        ntools.assert_true(mock_os_makedirs.called)
        ntools.assert_true(mock_osp_exists.called)
        mock_osp_exists.assert_called_once_with(dir_path)
        ntools.assert_equal(p, dir_path)

    @mock.patch('smqtk.utils.file_utils.os.path.exists')
    @mock.patch('smqtk.utils.file_utils.os.makedirs')
    def test_existError_noExist(self, mock_os_makedirs, mock_osp_exists):
        mock_os_makedirs.side_effect = OSError(errno.EEXIST,
                                               "Existing directory")
        mock_osp_exists.return_value = False

        dir_path = '/some/dir'
        ntools.assert_raises(OSError, file_utils.safe_create_dir, dir_path)

        mock_os_makedirs.assert_called_once_with(dir_path)
        mock_osp_exists.assert_called_once_with(dir_path)

    @mock.patch('smqtk.utils.file_utils.os.path.exists')
    @mock.patch('smqtk.utils.file_utils.os.makedirs')
    def test_otherOsError(self, mock_os_makedirs, mock_osp_exists):
        mock_os_makedirs.side_effect = OSError(errno.EACCES,
                                               "Permission Denied")

        dir_path = '/some/dir'
        ntools.assert_raises(OSError, file_utils.safe_create_dir, dir_path)

        mock_os_makedirs.assert_called_once_with(dir_path)
        ntools.assert_false(mock_osp_exists.called)

    @mock.patch('smqtk.utils.file_utils.os.makedirs')
    def test_otherException(self, mock_os_makedirs):
        mock_os_makedirs.side_effect = RuntimeError("Some other exception")

        dir_path = 'something'
        ntools.assert_raises(RuntimeError, file_utils.safe_create_dir, dir_path)

        mock_os_makedirs.assert_called_once_with(os.path.abspath(dir_path))
