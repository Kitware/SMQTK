from __future__ import print_function
import six

import mock
import nose.tools as ntools
import os
import unittest

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation.data_element import from_uri
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR


class TestDataFileElement (unittest.TestCase):

    def test_init_filepath_abs(self):
        fp = '/foo.txt'
        d = DataFileElement(fp)
        ntools.assert_equal(d._filepath, fp)

    def test_init_relFilepath_normal(self):
        # relative paths should be stored as given within the element
        fp = 'foo.txt'
        d = DataFileElement(fp)
        ntools.assert_equal(d._filepath, fp)

    def test_content_type(self):
        d = DataFileElement('foo.txt')
        ntools.assert_equal(d.content_type(), 'text/plain')

    def test_content_type_explicit_type(self):
        ex_type = 'image/png'
        d = DataFileElement('foo.txt', explicit_mimetype=ex_type)
        ntools.assert_equal(d.content_type(), ex_type)

    @mock.patch('smqtk.representation.data_element.DataElement.write_temp')
    def test_writeTempOverride(self, mock_DataElement_wt):
        # no manual directory, should return the base filepath
        expected_filepath = '/path/to/file.txt'
        d = DataFileElement(expected_filepath)
        fp = d.write_temp()

        ntools.assert_false(mock_DataElement_wt.called)
        ntools.assert_equal(expected_filepath, fp)

    @mock.patch('smqtk.representation.data_element.DataElement.write_temp')
    def test_writeTempOverride_sameDir(self, mock_DataElement_wt):
        expected_filepath = '/path/to/file.txt'
        target_dir = '/path/to'

        d = DataFileElement(expected_filepath)
        fp = d.write_temp(temp_dir=target_dir)

        ntools.assert_false(mock_DataElement_wt.called)
        ntools.assert_equal(fp, expected_filepath)

    @mock.patch('smqtk.representation.data_element.DataElement.write_temp')
    def test_writeTempOverride_diffDir(self, mock_DataElement_wt):
        """
        Test that adding ``temp_dir`` parameter triggers call to parent class
        """
        source_filepath = '/path/to/file.png'
        target_dir = '/some/other/dir'

        d = DataFileElement(source_filepath)

        # Should call parent class write_temp since target is not the same dir
        # that the source file is in.
        mock_DataElement_wt.return_value = 'expected'
        v = d.write_temp(temp_dir=target_dir)
        ntools.assert_equal(v, 'expected')
        mock_DataElement_wt.assert_called_with(target_dir)

    def test_cleanTemp(self):
        # a write temp and clean temp should not affect original file
        source_file = os.path.join(TEST_DATA_DIR, 'test_file.dat')
        ntools.assert_true(os.path.isfile(source_file))
        d = DataFileElement(source_file)
        d.write_temp()
        ntools.assert_equal(len(d._temp_filepath_stack), 0)
        d.clean_temp()
        ntools.assert_true(os.path.isfile(source_file))

    def test_fromConfig(self):
        fp = os.path.join(TEST_DATA_DIR, "Lenna.png")
        c = {
            "filepath": fp
        }
        df = DataFileElement.from_config(c)
        ntools.assert_equal(df._filepath, fp)

    def test_toConfig(self):
        fp = os.path.join(TEST_DATA_DIR, "Lenna.png")
        df = DataFileElement(fp)
        c = df.get_config()
        ntools.assert_equal(c['filepath'], fp)

    def test_configuration(self):
        fp = os.path.join(TEST_DATA_DIR, "Lenna.png")
        default_config = DataFileElement.get_default_config()
        ntools.assert_equal(default_config,
                            {'filepath': None, 'readonly': False,
                             'explicit_mimetype': None})

        default_config['filepath'] = fp
        inst1 = DataFileElement.from_config(default_config)
        ntools.assert_equal(default_config, inst1.get_config())

        inst2 = DataFileElement.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)

    def test_repr(self):
        e = DataFileElement('foo')
        ntools.assert_equal(repr(e),
                            "DataFileElement{filepath: foo, readonly: False, "
                            "explicit_mimetype: None}")

        e = DataFileElement('bar', readonly=True)
        ntools.assert_equal(repr(e),
                            "DataFileElement{filepath: bar, readonly: True, "
                            "explicit_mimetype: None}")

        e = DataFileElement('baz', readonly=True, explicit_mimetype='some/type')
        ntools.assert_equal(repr(e),
                            "DataFileElement{filepath: baz, readonly: True, "
                            "explicit_mimetype: some/type}")

    def test_from_uri_invalid_uri_empty(self):
        # Given empty string
        ntools.assert_raises(
            InvalidUriError,
            DataFileElement.from_uri,
            ''
        )

    def test_from_uri_invalid_uri_malformed_rel_directory(self):
        # URI malformed: relative path trailing slash (directory)
        ntools.assert_raises(
            InvalidUriError,
            DataFileElement.from_uri,
            "some/rel/path/dir/"
        )

    def test_from_uri_invalid_uri_malformed_abs_directory(self):
        # URI malformed: absolute path trailing slash (directory)
        ntools.assert_raises(
            InvalidUriError,
            DataFileElement.from_uri,
            "/abs/path/dir/"
        )

    def test_from_uri_invalid_uri_malformed_bad_header(self):
        # URI malformed: file:// malformed

        # Missing colon
        ntools.assert_raises(
            InvalidUriError,
            DataFileElement.from_uri,
            "file///some/file/somewhere.txt"
        )

        # file misspelled
        ntools.assert_raises(
            InvalidUriError,
            DataFileElement.from_uri,
            "fle:///some/file/somewhere.txt"
        )

    def test_from_uri_invalid_uri_malformed_header_rel_path(self):
        # URL malformed: file:// not given ABS path
        ntools.assert_raises(
            InvalidUriError,
            DataFileElement.from_uri,
            "file://some/rel/path.txt"
        )

    # noinspection PyUnresolvedReferences
    def test_from_uri(self):
        # will be absolute path
        test_file_path = os.path.join(TEST_DATA_DIR, "test_file.dat")
        print("Test file path:", test_file_path)

        e = DataFileElement.from_uri(test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), six.b(''))

        e = DataFileElement.from_uri('file://' + test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), six.b(''))

    # noinspection PyUnresolvedReferences
    def test_from_uri_plugin_level(self):
        # will be absolute path
        test_file_path = os.path.join(TEST_DATA_DIR, "test_file.dat")
        print("Test file path:", test_file_path)

        e = from_uri(test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), six.b(''))

        e = from_uri('file://' + test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), six.b(''))

    def test_is_empty_file_not_exists(self):
        e = DataFileElement('/no/exists')
        ntools.assert_true(e.is_empty())

    def test_is_empty_file_zero_data(self):
        e = DataFileElement(os.path.join(TEST_DATA_DIR, 'test_file.dat'))
        ntools.assert_true(e.is_empty())

    def test_is_empty_file_has_data(self):
        e = DataFileElement(os.path.join(TEST_DATA_DIR, 'Lenna.png'))
        ntools.assert_false(e.is_empty())

    def test_get_bytes_no_file(self):
        e = DataFileElement("/not/a/valid/path.txt", readonly=True)
        # We currently expect, in the case where the filepath doesn't exist, to
        # get the same bytes as if the file existed and were empty.
        self.assertEqual(e.get_bytes(), six.b(""))
        # read-only status should have no effect.
        e = DataFileElement("/not/a/valid/path.txt", readonly=True)
        self.assertEqual(e.get_bytes(), six.b(""))

    def test_get_bytes(self):
        # Test with a known real file.
        test_file_path = os.path.join(TEST_DATA_DIR, 'text_file')
        e = DataFileElement(test_file_path)
        self.assertEqual(e.get_bytes(), six.b("Some text content.\n"))

    def test_writable_readonly_false(self):
        e = DataFileElement('foo')
        ntools.assert_true(e.writable())

        e = DataFileElement('foo', False)
        ntools.assert_true(e.writable())

        e = DataFileElement('foo', readonly=False)
        ntools.assert_true(e.writable())

    def test_writable_readonly_true(self):
        e = DataFileElement('foo', True)
        ntools.assert_false(e.writable())

        e = DataFileElement('foo', readonly=True)
        ntools.assert_false(e.writable())

    @mock.patch('smqtk.representation.data_element.file_element.safe_file_write')
    def test_set_bytes_writable(self, m_sfw):
        # Using a relative filepath
        test_path = 'foo'
        test_bytes = six.b('test string of bytes')

        e = DataFileElement(test_path)
        e.set_bytes(test_bytes)

        # File write function should be called
        m_sfw.assert_called_once_with(test_path, test_bytes)

    def test_set_bytes_readonly(self):
        e = DataFileElement('foo', readonly=True)
        ntools.assert_raises(
            ReadOnlyError,
            e.set_bytes,
            six.b('some bytes')
        )
