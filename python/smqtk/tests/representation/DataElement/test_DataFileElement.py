import mock
import nose.tools as ntools
import os
import unittest

from smqtk.exceptions import InvalidUriError
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
                            {'filepath': None, 'readonly': False})

        default_config['filepath'] = fp
        inst1 = DataFileElement.from_config(default_config)
        ntools.assert_equal(default_config, inst1.get_config())

        inst2 = DataFileElement.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)

    def test_repr(self):
        e = DataFileElement('foo')
        ntools.assert_equal(repr(e),
                            "DataFileElement{filepath: foo, readonly: False}")

        e = DataFileElement('bar', readonly=True)
        ntools.assert_equal(repr(e),
                            "DataFileElement{filepath: bar, readonly: True}")

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
        print "Test file path:", test_file_path

        e = DataFileElement.from_uri(test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), '')

        e = DataFileElement.from_uri('file://' + test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), '')

    # noinspection PyUnresolvedReferences
    def test_from_uri_plugin_level(self):
        # will be absolute path
        test_file_path = os.path.join(TEST_DATA_DIR, "test_file.dat")
        print "Test file path:", test_file_path

        e = from_uri(test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), '')

        e = from_uri('file://' + test_file_path)
        ntools.assert_is_instance(e, DataFileElement)
        ntools.assert_equal(e._filepath, test_file_path)
        ntools.assert_equal(e.get_bytes(), '')
