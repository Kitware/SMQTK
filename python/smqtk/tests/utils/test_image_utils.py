import mock
import os
import sys
import tempfile
import unittest

from six.moves import StringIO

from smqtk.bin.check_images import main as check_images_main
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR
from smqtk.utils.image_utils import is_loadable_image, is_valid_element


class TestIsLoadableImage(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.good_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                      'Lenna.png'))
        cls.non_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                     'test_file.dat'))

    def test_non_data_element_raises_exception(self):
        # should throw:
        # AttributeError: 'bool' object has no attribute 'get_bytes'
        self.assertRaises(
            AttributeError,
            is_loadable_image, False
        )

    def test_unloadable_image_returns_false(self):
        assert is_loadable_image(self.non_image) is False

    def test_loadable_image_returns_true(self):
        assert is_loadable_image(self.good_image) is True


class TestIsValidElement(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.good_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                      'Lenna.png'))
        cls.non_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                     'test_file.dat'))

    def test_non_data_element(self):
        # Should check that input datum is a DataElement instance.
        # noinspection PyTypeChecker
        assert is_valid_element(False) is False

    def test_invalid_content_type(self):
        assert is_valid_element(self.good_image, valid_content_types=[]) \
               is False

    def test_valid_content_type(self):
        assert is_valid_element(self.good_image,
                                valid_content_types=['image/png']) is True

    def test_invalid_image_returns_false(self):
        assert is_valid_element(self.non_image, check_image=True) is False


class TestCheckImageCli(unittest.TestCase):

    @staticmethod
    def check_images():
        """ Simulate execution of check_images utility main. """
        saved_stdout, saved_stderr = sys.stdout, sys.stderr

        out, err = StringIO(), StringIO()
        try:
            sys.stdout, sys.stderr = out, err
            check_images_main()
        except SystemExit as ex:
            print("Encountered SystemExit exception, code {}".format(ex.code))
        finally:
            stdout, stderr = out.getvalue().strip(), err.getvalue().strip()
            sys.stdout, sys.stderr = saved_stdout, saved_stderr

        return stdout, stderr

    def test_base_case(self):
        # noinspection PyUnresolvedReferences
        with mock.patch.object(sys, 'argv', ['']):
            assert 'Validate a list of images returning the filepaths' in \
                self.check_images()[0]

    def test_check_images(self):
        # Create test file with a valid, invalid, and non-existent image
        _, filename = tempfile.mkstemp()

        with open(filename, 'w') as outfile:
            outfile.write(os.path.join(TEST_DATA_DIR, 'Lenna.png') + '\n')
            outfile.write(os.path.join(TEST_DATA_DIR, 'test_file.dat') + '\n')
            outfile.write(os.path.join(TEST_DATA_DIR, 'non-existent-file.jpeg'))

        # noinspection PyUnresolvedReferences
        with mock.patch.object(sys, 'argv', ['', '--file-list', filename]):
            out, err = self.check_images()

            assert out == ','.join([os.path.join(TEST_DATA_DIR, 'Lenna.png'),
                                    '3ee0d360dc12003c0d43e3579295b52b64906e85'])
            assert 'non-existent-file.jpeg' not in out

        # noinspection PyUnresolvedReferences
        with mock.patch.object(sys, 'argv',
                               ['', '--file-list', filename, '--invert']):
            out, err = self.check_images()

            assert out == ','.join([os.path.join(TEST_DATA_DIR,
                                                 'test_file.dat'),
                                    'da39a3ee5e6b4b0d3255bfef95601890afd80709'])
            assert 'non-existent-file.jpeg' not in out
