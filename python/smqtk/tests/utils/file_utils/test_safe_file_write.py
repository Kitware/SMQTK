import mock
import unittest

import nose.tools

from smqtk.utils.file_utils import safe_file_write


class TestSafeFileWrite (unittest.TestCase):
    """
    Tests for the ``smqtk.utils.file_utils.safe_file_write`` function.

    Mocking out underlying function that would have filesystem side effects.
    """

    @mock.patch('smqtk.utils.file_utils.safe_create_dir')
    @mock.patch('smqtk.utils.file_utils.os.rename')
    @mock.patch('smqtk.utils.file_utils.os.close')
    @mock.patch('smqtk.utils.file_utils.os.remove')
    @mock.patch('smqtk.utils.file_utils.os.write')
    @mock.patch('smqtk.utils.file_utils.tempfile.mkstemp')
    def test_safe_file_write_relative_simple(
            self, m_mkstemp, m_write, m_remove, m_close, m_rename, m_scd):
        # Experimental filepath and content.
        fp = 'bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        test_tmp_fd = 'temp fd'
        test_tmp_fp = 'temp fp'
        m_mkstemp.return_value = (test_tmp_fd, test_tmp_fp)
        # Mock return from write simulating complete byte writing.
        m_write.return_value = len(expected_bytes)

        safe_file_write(fp, expected_bytes)

        m_scd.assert_called_once_with('')
        m_mkstemp.assert_called_once_with(suffix='.txt', prefix='bar.', dir='')
        m_write.assert_called_once_with(test_tmp_fd, expected_bytes)
        nose.tools.assert_equal(m_remove.call_count, 0)
        m_close.assert_called_once_with(test_tmp_fd)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file_utils.safe_create_dir')
    @mock.patch('smqtk.utils.file_utils.os.rename')
    @mock.patch('smqtk.utils.file_utils.os.close')
    @mock.patch('smqtk.utils.file_utils.os.remove')
    @mock.patch('smqtk.utils.file_utils.os.write')
    @mock.patch('smqtk.utils.file_utils.tempfile.mkstemp')
    def test_safe_file_write_relative_subdir(
            self, m_mkstemp, m_write, m_remove, m_close, m_rename, m_scd):
        # Experimental filepath and content.
        fp = 'foo/other/bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        test_tmp_fd = 'temp fd'
        test_tmp_fp = 'temp fp'
        m_mkstemp.return_value = (test_tmp_fd, test_tmp_fp)
        # Mock return from write simulating complete byte writing.
        m_write.return_value = len(expected_bytes)

        safe_file_write(fp, expected_bytes)

        m_scd.assert_called_once_with('foo/other')
        m_mkstemp.assert_called_once_with(suffix='.txt', prefix='bar.',
                                          dir='foo/other')
        m_write.assert_called_once_with(test_tmp_fd, expected_bytes)
        nose.tools.assert_equal(m_remove.call_count, 0)
        m_close.assert_called_once_with(test_tmp_fd)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file_utils.safe_create_dir')
    @mock.patch('smqtk.utils.file_utils.os.rename')
    @mock.patch('smqtk.utils.file_utils.os.close')
    @mock.patch('smqtk.utils.file_utils.os.remove')
    @mock.patch('smqtk.utils.file_utils.os.write')
    @mock.patch('smqtk.utils.file_utils.tempfile.mkstemp')
    def test_safe_file_write_custom_tmp_dir(
            self, m_mkstemp, m_write, m_remove, m_close, m_rename, m_scd):
        # Experimental filepath and content.
        fp = 'foo/other/bar.txt'
        expected_bytes = 'hello world'
        custom_tmp_dir = '/some/other/directory'

        # Mock return for temp file creation so we can check os.* calls.
        test_tmp_fd = 'temp fd'
        test_tmp_fp = 'temp fp'
        m_mkstemp.return_value = (test_tmp_fd, test_tmp_fp)
        # Mock return from write simulating complete byte writing.
        m_write.return_value = len(expected_bytes)

        safe_file_write(fp, expected_bytes, custom_tmp_dir)

        m_scd.assert_called_once_with('foo/other')
        m_mkstemp.assert_called_once_with(suffix='.txt', prefix='bar.',
                                          dir=custom_tmp_dir)
        m_write.assert_called_once_with(test_tmp_fd, expected_bytes)
        nose.tools.assert_equal(m_remove.call_count, 0)
        m_close.assert_called_once_with(test_tmp_fd)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file_utils.safe_create_dir')
    @mock.patch('smqtk.utils.file_utils.os.rename')
    @mock.patch('smqtk.utils.file_utils.os.close')
    @mock.patch('smqtk.utils.file_utils.os.remove')
    @mock.patch('smqtk.utils.file_utils.os.write')
    @mock.patch('smqtk.utils.file_utils.tempfile.mkstemp')
    def test_safe_file_write_absolute(
            self, m_mkstemp, m_write, m_remove, m_close, m_rename, m_scd):
        # Experimental filepath and content.
        fp = '/some/absolute/dir/bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        test_tmp_fd = 'temp fd'
        test_tmp_fp = 'temp fp'
        m_mkstemp.return_value = (test_tmp_fd, test_tmp_fp)
        # Mock return from write simulating complete byte writing.
        m_write.return_value = len(expected_bytes)

        safe_file_write(fp, expected_bytes)

        m_scd.assert_called_once_with('/some/absolute/dir')
        m_mkstemp.assert_called_once_with(suffix='.txt', prefix='bar.',
                                          dir='/some/absolute/dir')
        m_write.assert_called_once_with(test_tmp_fd, expected_bytes)
        nose.tools.assert_equal(m_remove.call_count, 0)
        m_close.assert_called_once_with(test_tmp_fd)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file_utils.safe_create_dir')
    @mock.patch('smqtk.utils.file_utils.os.rename')
    @mock.patch('smqtk.utils.file_utils.os.close')
    @mock.patch('smqtk.utils.file_utils.os.remove')
    @mock.patch('smqtk.utils.file_utils.os.write')
    @mock.patch('smqtk.utils.file_utils.tempfile.mkstemp')
    def test_safe_file_write_invalid_write_return(
            self, m_mkstemp, m_write, m_remove, m_close, m_rename, m_scd):
        # Test for what happens when os.write does not respond with all bytes
        # written.

        # Experimental filepath and content.
        fp = 'bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        test_tmp_fd = 'temp fd'
        test_tmp_fp = 'temp fp'
        m_mkstemp.return_value = (test_tmp_fd, test_tmp_fp)
        # Mock return from write simulating not all bytes being written.
        m_write.return_value = len(expected_bytes) - 3

        nose.tools.assert_raises(
            RuntimeError,
            safe_file_write, fp, expected_bytes
        )

        m_scd.assert_called_once_with('')
        m_mkstemp.assert_called_once_with(suffix='.txt', prefix='bar.', dir='')
        m_write.assert_called_once_with(test_tmp_fd, expected_bytes)
        # Remove should now be called on temp file path
        nose.tools.assert_equal(m_remove.call_count, 1)
        m_remove.assert_called_once_with(test_tmp_fp)
        m_close.assert_called_once_with(test_tmp_fd)
        # Rename should no longer be called.
        nose.tools.assert_equal(m_rename.call_count, 0)
