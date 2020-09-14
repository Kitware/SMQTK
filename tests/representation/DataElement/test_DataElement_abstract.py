"""
Tests for DataElement abstract interface class methods that provide
functionality.
"""
import hashlib
import unittest.mock as mock
import os.path as osp
import tempfile
import unittest
import six

import smqtk.exceptions
import smqtk.representation.data_element

if six.PY2:
    builtin_open = '__builtin__.open'
else:
    builtin_open = 'builtins.open'


# because this has a stable mimetype conversion
EXPECTED_CONTENT_TYPE = "image/png"
EXPECTED_BYTES = six.b("hello world")
EXPECTED_MD5 = hashlib.md5(EXPECTED_BYTES).hexdigest()
EXPECTED_SHA1 = hashlib.sha1(EXPECTED_BYTES).hexdigest()
EXPECTED_SHA512 = hashlib.sha512(EXPECTED_BYTES).hexdigest()
# UUID is currently set to be equivalent to SHA1 value by default
EXPECTED_UUID = EXPECTED_SHA1


# Caches the temp directory before we start mocking things out that would
# otherwise be required for the tempfile module to determine the temp
# directory.
tempfile.gettempdir()


# noinspection PyClassHasNoInit,PyAbstractClass
class DummyDataElement (smqtk.representation.data_element.DataElement):

    TEST_WRITABLE = True
    TEST_BYTES = EXPECTED_BYTES
    TEST_CONTENT_TYPE = EXPECTED_CONTENT_TYPE

    @classmethod
    def is_usable(cls):
        return True

    def __repr__(self):
        return super(DummyDataElement, self).__repr__()

    def get_config(self):
        return {}

    def content_type(self):
        return self.TEST_CONTENT_TYPE

    def is_empty(self):
        pass

    def get_bytes(self):
        return self.TEST_BYTES

    def set_bytes(self, b):
        super(DummyDataElement, self).set_bytes(b)
        self.TEST_BYTES = b

    def writable(self):
        return self.TEST_WRITABLE


class TestDataElementAbstract (unittest.TestCase):

    def test_from_uri_default(self):
        self.assertRaises(
            smqtk.exceptions.NoUriResolutionError,
            DummyDataElement.from_uri, 'some uri'
        )

    def test_not_hashable(self):
        # Hash should be that of the UUID of the element
        de = DummyDataElement()
        self.assertRaises(TypeError, hash, de)

    def test_del(self):
        de = DummyDataElement()
        m_clean_temp = de.clean_temp = mock.Mock()
        del de

        self.assertTrue(m_clean_temp.called)

    def test_equality(self):
        # equal when binary content is the same
        e1 = DummyDataElement()
        e2 = DummyDataElement()

        test_content_1 = 'some similar content'
        e1.TEST_BYTES = e2.TEST_BYTES = test_content_1
        self.assertEqual(e1, e2)

        test_content_2 = 'some other bytes'
        e2.TEST_BYTES = test_content_2
        self.assertNotEqual(e1, e2)

    def test_md5(self):
        de = DummyDataElement()
        md5 = de.md5()
        self.assertEqual(md5, EXPECTED_MD5)

    def test_sha1(self):
        de = DummyDataElement()
        sha1 = de.sha1()
        self.assertEqual(sha1, EXPECTED_SHA1)

    def test_sha512(self):
        de = DummyDataElement()
        sha1 = de.sha512()
        self.assertEqual(sha1, EXPECTED_SHA512)

    @mock.patch('smqtk.representation.data_element.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch(builtin_open)
    def test_content_type_extension(self,
                                    _mock_open, _mock_os_open, _mock_os_close,
                                    _mock_fcntl, _mock_scd):
        de = DummyDataElement()
        de.content_type = mock.Mock(return_value=None)
        fname = de.write_temp()
        self.assertFalse(fname.endswith('.png'))

        fname = DummyDataElement().write_temp()
        self.assertTrue(fname.endswith('.png'))

    # Cases:
    #   - no existing temps, no specific dir
    #   - no existing temps, given specific dir
    #   - existing temps, no specific dir
    #   - existing temps, given specific dir
    #
    # Mocking open, os.open, os.close and fcntl to actual file interaction
    #   - os.open is used under the hood of tempfile to open a file (which also
    #       creates it on disk).

    @mock.patch('smqtk.representation.data_element.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch(builtin_open)
    def test_writeTemp_noExisting_noDir(self,
                                        mock_open, _mock_os_open,
                                        _mock_os_close, _mock_fcntl, mock_scd):
        # no existing temps, no specific dir
        fp = DummyDataElement().write_temp()

        self.assertFalse(mock_scd.called)
        self.assertTrue(mock_open.called)
        self.assertEqual(osp.dirname(fp), tempfile.gettempdir())

    @mock.patch('smqtk.representation.data_element.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch(builtin_open)
    def test_writeTemp_noExisting_givenDir(self,
                                           mock_open, _mock_os_open,
                                           _mock_os_close, _mock_fcntl,
                                           mock_scd):
        # no existing temps, given specific dir
        target_dir = '/some/dir/somewhere'

        fp = DummyDataElement().write_temp(target_dir)

        mock_scd.assert_called_once_with(target_dir)
        self.assertTrue(mock_open.called)
        self.assertNotEqual(osp.dirname(fp), tempfile.gettempdir())
        self.assertEqual(osp.dirname(fp), target_dir)

    @mock.patch("smqtk.representation.data_element.file_element.osp.isfile")
    @mock.patch('smqtk.representation.data_element.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch(builtin_open)
    def test_writeTemp_hasExisting_noDir(self,
                                         mock_open, _mock_os_open,
                                         _mock_os_close, _mock_fcntl, mock_scd,
                                         mock_isfile):
        # Pretend we have existing temps. Will to "write" a temp file to no
        # specific dir, which should not write anything new and just return the
        # last path in the list.
        prev_0 = '/tmp/file.txt'
        prev_1 = '/tmp/file_two.png'

        de = DummyDataElement()
        de._temp_filepath_stack.append(prev_0)
        de._temp_filepath_stack.append(prev_1)

        # Make sure os.path.isfile returns true so we think things in temp
        # stack exist.
        def osp_isfile_se(path):
            if simulate and path in {prev_0, prev_1}:
                return True
            else:
                return False
        simulate = True
        mock_isfile.side_effect = osp_isfile_se

        fp = de.write_temp()

        self.assertFalse(mock_scd.called)
        self.assertFalse(mock_open.called)
        self.assertEqual(fp, prev_1)

        # _temp_filepath_stack files don't exist, so make sure isfile returns
        # false so clean_temp doesn't try to remove files that don't exist.
        simulate = False

    @mock.patch('smqtk.representation.data_element.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch(builtin_open)
    def test_writeTemp_hasExisting_givenNewDir(self, mock_open, _mock_os_open,
                                               _mock_os_close, _mock_fcntl,
                                               mock_scd):
        # existing temps, given specific dir
        prev_0 = '/tmp/file.txt'
        prev_1 = '/tmp/file_two.png'

        target_dir = '/some/specific/dir'

        de = DummyDataElement()
        de._temp_filepath_stack.append(prev_0)
        de._temp_filepath_stack.append(prev_1)

        fp = de.write_temp(temp_dir=target_dir)

        self.assertTrue(mock_scd.called)
        self.assertTrue(mock_open.called)
        self.assertEqual(osp.dirname(fp), target_dir)

    @mock.patch("smqtk.representation.data_element.file_element.osp.isfile")
    @mock.patch('smqtk.representation.data_element.safe_create_dir')
    @mock.patch('fcntl.fcntl')  # global
    @mock.patch('os.close')  # global
    @mock.patch('os.open')  # global
    @mock.patch(builtin_open)
    def test_writeTemp_hasExisting_givenExistingDir(self, mock_open,
                                                    _mock_os_open,
                                                    _mock_os_close,
                                                    _mock_fcntl,
                                                    mock_scd, mock_isfile):
        # Pretend these files already exist as written temp files.
        # We test that write_temp with a target directory yields a previously
        #   "written" temp file.
        #
        # that given specific dir already in stack
        prev_0 = '/dir1/file.txt'
        prev_1 = '/tmp/things/file_two.png'
        prev_2 = '/some/specific/dir'

        def osp_isfile_se(path):
            if simulate and path in {prev_0, prev_1, prev_2}:
                return True
            else:
                return False
        simulate = True
        mock_isfile.side_effect = osp_isfile_se

        de = DummyDataElement()
        de._temp_filepath_stack.append(prev_0)
        de._temp_filepath_stack.append(prev_1)
        de._temp_filepath_stack.append(prev_2)

        target_dir = "/tmp/things"

        # Make sure os.path.isfile returns true so we think things in temp
        # stack exist.
        mock_isfile.return_value = True

        fp = de.write_temp(temp_dir=target_dir)

        self.assertFalse(mock_scd.called)
        self.assertFalse(mock_open.called)
        self.assertEqual(fp, prev_1)

        # _temp_filepath_stack files don't exist, so make sure isfile returns
        # false so clean_temp doesn't try to remove files that don't exist.
        simulate = False

    @mock.patch("smqtk.representation.data_element.os")
    def test_cleanTemp_noTemp(self, mock_os):
        # should do all of nothing
        de = DummyDataElement()

        de.clean_temp()

        self.assertFalse(mock_os.path.isfile.called)
        self.assertFalse(mock_os.remove.called)

    @mock.patch("smqtk.representation.data_element.os")
    def test_cleanTemp_hasTemp_badPath(self, mock_os):
        de = DummyDataElement()
        de._temp_filepath_stack.append('tmp/thing')
        mock_os.path.isfile.return_value = False

        de.clean_temp()

        mock_os.path.isfile.assert_called_once_with('tmp/thing')
        self.assertFalse(mock_os.remove.called)

    @mock.patch("smqtk.representation.data_element.os")
    def test_cleanTemp_hasTemp_validPath(self, mock_os):
        expected_path = '/tmp/something'

        de = DummyDataElement()
        de._temp_filepath_stack.append(expected_path)
        mock_os.path.isfile.return_value = True

        de.clean_temp()

        mock_os.path.isfile.assert_called_once_with(expected_path)
        mock_os.remove.assert_called_once_with(expected_path)

    def test_uuid(self):
        de = DummyDataElement()
        de.TEST_BYTES = EXPECTED_BYTES
        self.assertEqual(de.uuid(), EXPECTED_UUID)

    def test_to_buffered_reader(self):
        # Check that we get expected file-like returns.
        de = DummyDataElement()
        de.TEST_BYTES = EXPECTED_BYTES
        br = de.to_buffered_reader()
        self.assertEqual(br.readlines(), [six.b('hello world')])

        de.TEST_BYTES = six.b('some content\nwith new \nlines')
        br = de.to_buffered_reader()
        self.assertEqual(br.readlines(),
                         [six.b('some content\n'),
                          six.b('with new \n'),
                          six.b('lines')])

    def test_is_read_only(self):
        de = DummyDataElement()
        de.TEST_WRITABLE = True
        self.assertFalse(de.is_read_only())
        de.TEST_WRITABLE = False
        self.assertTrue(de.is_read_only())

    def test_set_bytes_not_writable(self):
        de = DummyDataElement()
        # trigger UUID cache at least once
        self.assertEqual(de.uuid(), EXPECTED_UUID)

        de.TEST_WRITABLE = False
        self.assertRaises(
            smqtk.exceptions.ReadOnlyError,
            de.set_bytes, six.b('test bytes')
        )

        # Caches shouldn't have been invalidated due to error
        self.assertEqual(de.uuid(), EXPECTED_UUID)

    def test_set_bytes_checksum_cache_invalidation(self):
        de = DummyDataElement()
        # trigger UUID cache at least once
        self.assertEqual(de.uuid(), EXPECTED_UUID)

        new_expected_bytes = six.b('some new byte content')
        new_expected_uuid = hashlib.sha1(new_expected_bytes).hexdigest()

        de.TEST_WRITABLE = True
        de.set_bytes(new_expected_bytes)

        # Caches should have been invalidated, so UUID return should now
        # reflect new byte content.
        self.assertNotEqual(de.uuid(), EXPECTED_UUID)
        self.assertEqual(de.uuid(), new_expected_uuid)
