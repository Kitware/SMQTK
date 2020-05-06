import unittest.mock as mock
import unittest

from smqtk.utils.file import safe_file_write


class TestSafeFileWrite (unittest.TestCase):
    """
    Tests for the ``smqtk.utils.file.safe_file_write`` function.

    Mocking out underlying function that would have filesystem side effects.
    """

    @mock.patch('smqtk.utils.file.safe_create_dir')
    @mock.patch('smqtk.utils.file.os.rename')
    @mock.patch('smqtk.utils.file.os.remove')
    @mock.patch('smqtk.utils.file.tempfile.NamedTemporaryFile')
    def test_safe_file_write_relative_simple(
            self, m_NTF, m_remove, m_rename, m_scd):
        # Experimental filepath and content.
        fp = 'bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        m_file = m_NTF.return_value
        test_tmp_fp = 'temp fp'
        m_file.name = test_tmp_fp

        safe_file_write(fp, expected_bytes)

        m_scd.assert_called_once_with('')
        m_NTF.assert_called_once_with(suffix='.txt', prefix='bar.', dir='',
                                      delete=False)
        m_file.write.assert_called_once_with(expected_bytes)
        m_file.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(m_remove.call_count, 0)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file.safe_create_dir')
    @mock.patch('smqtk.utils.file.os.rename')
    @mock.patch('smqtk.utils.file.os.remove')
    @mock.patch('smqtk.utils.file.tempfile.NamedTemporaryFile')
    def test_safe_file_write_relative_subdir(
            self, m_NTF, m_remove, m_rename, m_scd):
        # Experimental filepath and content.
        fp = 'foo/other/bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        m_file = m_NTF.return_value
        test_tmp_fp = 'temp fp'
        m_file.name = test_tmp_fp

        safe_file_write(fp, expected_bytes)

        m_scd.assert_called_once_with('foo/other')
        m_NTF.assert_called_once_with(suffix='.txt', prefix='bar.',
                                      dir='foo/other', delete=False)
        m_file.write.assert_called_once_with(expected_bytes)
        m_file.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(m_remove.call_count, 0)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file.safe_create_dir')
    @mock.patch('smqtk.utils.file.os.rename')
    @mock.patch('smqtk.utils.file.os.remove')
    @mock.patch('smqtk.utils.file.tempfile.NamedTemporaryFile')
    def test_safe_file_write_custom_tmp_dir(
            self, m_NTF, m_remove, m_rename, m_scd):
        # Experimental filepath and content.
        fp = 'foo/other/bar.txt'
        expected_bytes = 'hello world'
        custom_tmp_dir = '/some/other/directory'

        # Mock return for temp file creation so we can check os.* calls.
        m_file = m_NTF.return_value
        test_tmp_fp = 'temp fp'
        m_file.name = test_tmp_fp

        safe_file_write(fp, expected_bytes, custom_tmp_dir)

        m_scd.assert_called_once_with('foo/other')
        m_NTF.assert_called_once_with(suffix='.txt', prefix='bar.',
                                      dir=custom_tmp_dir, delete=False)
        m_file.write.assert_called_once_with(expected_bytes)
        m_file.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(m_remove.call_count, 0)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file.safe_create_dir')
    @mock.patch('smqtk.utils.file.os.rename')
    @mock.patch('smqtk.utils.file.os.remove')
    @mock.patch('smqtk.utils.file.tempfile.NamedTemporaryFile')
    def test_safe_file_write_absolute(
            self, m_NTF, m_remove, m_rename, m_scd):
        # Experimental filepath and content.
        fp = '/some/absolute/dir/bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        m_file = m_NTF.return_value
        test_tmp_fp = 'temp fp'
        m_file.name = test_tmp_fp

        safe_file_write(fp, expected_bytes)

        m_scd.assert_called_once_with('/some/absolute/dir')
        m_NTF.assert_called_once_with(suffix='.txt', prefix='bar.',
                                      dir='/some/absolute/dir', delete=False)
        m_file.write.assert_called_once_with(expected_bytes)
        m_file.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(m_remove.call_count, 0)
        m_rename.assert_called_once_with(test_tmp_fp, fp)

    @mock.patch('smqtk.utils.file.safe_create_dir')
    @mock.patch('smqtk.utils.file.os.rename')
    @mock.patch('smqtk.utils.file.os.remove')
    @mock.patch('smqtk.utils.file.tempfile.NamedTemporaryFile')
    def test_safe_file_write_raising_write(
            self, m_NTF, m_remove, m_rename, m_scd):
        # Test for what happens when file.write raises an exception.

        # Experimental filepath and content.
        fp = 'bar.txt'
        expected_bytes = 'hello world'

        # Mock return for temp file creation so we can check os.* calls.
        m_file = m_NTF.return_value
        test_tmp_fp = 'temp fp'
        m_file.name = test_tmp_fp
        # Mock return from write simulating not all bytes being written.
        m_file.write.side_effect = OSError

        self.assertRaises(
            OSError,
            safe_file_write, fp, expected_bytes
        )

        m_scd.assert_called_once_with('')
        m_NTF.assert_called_once_with(suffix='.txt', prefix='bar.', dir='',
                                      delete=False)
        m_file.write.assert_called_once_with(expected_bytes)
        # Remove should now be called on temp file path
        self.assertEqual(m_remove.call_count, 1)
        m_remove.assert_called_once_with(test_tmp_fp)
        self.assertEqual(m_file.__exit__.call_count, 1)
        # Rename should no longer be called.
        self.assertEqual(m_rename.call_count, 0)
