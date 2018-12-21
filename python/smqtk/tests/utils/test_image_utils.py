import mock
import os
import sys
import tempfile
import unittest

from six.moves import StringIO

from smqtk.bin.check_images import main as check_images_main
from smqtk.representation import AxisAlignedBoundingBox
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR

from smqtk.utils.image import (
    is_loadable_image,
    is_valid_element,
    crop_in_bounds,
)


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


class TestCropInBounds(object):
    """
    Test using the ``crop_in_bounds`` function.
    """

    def test_in_bounds_inside(self):
        """
        Test that ``in_bounds`` passes when crop inside given rectangle bounds.

            +--+
            |  |
            |##|  => (4, 6) image, (2,2) crop
            |##|
            |  |
            +--+

        """
        bb = AxisAlignedBoundingBox([1, 2], [3, 4])
        assert crop_in_bounds(bb, 4, 8)

    def test_in_bounds_inside_edges(self):
        """
        Test that a crop is "in bounds" when contacting the 4 edges of the
        given rectangular bounds.

            +--+
            |  |
            ## |  => (4, 6) image, (2,2) crop
            ## |
            |  |
            +--+

            +##+
            |##|
            |  |  => (4, 6) image, (2,2) crop
            |  |
            |  |
            +--+

            +--+
            |  |
            | ##  => (4, 6) image, (2,2) crop
            | ##
            |  |
            +--+

            +--+
            |  |
            |  |  => (4, 6) image, (2,2) crop
            |  |
            |##|
            +##+

        """
        # noinspection PyArgumentList
        bb = AxisAlignedBoundingBox([0, 2], [2, 4])
        assert crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, 0], [3, 2])
        assert crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([2, 2], [4, 4])
        assert crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, 4], [3, 6])
        assert crop_in_bounds(bb, 4, 6)

    def test_in_bounds_completely_outside(self):
        """
        Test that being completely outside the given bounds causes
        ``in_bounds`` to return False.
        """
        bb = AxisAlignedBoundingBox([100, 100], [102, 102])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([-100, -100], [-98, -98])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([-100, 100], [-98, 102])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([100, -100], [102, -98])
        assert not crop_in_bounds(bb, 4, 6)

    def test_in_bounds_crossing_edges(self):
        """
        Test that ``in_bounds`` returns False when crop crossed the 4 edges.

            +--+
            |  |
           ### |  => (4, 6) image, (3,2) crop
           ### |
            |  |
            +--+

            +--+
            |  |
            | ###  => (4, 6) image, (3,2) crop
            | ###
            |  |
            +--+

             ##
            +##+
            |##|
            |  |  => (4, 6) image, (2,3) crop
            |  |
            |  |
            +--+

            +--+
            |  |
            |  |  => (4, 6) image, (2,3) crop
            |  |
            |##|
            +##+
             ##

        """
        bb = AxisAlignedBoundingBox([-1, 2], [2, 4])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([2, 2], [5, 4])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, -1], [3, 2])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, 4], [3, 7])
        assert not crop_in_bounds(bb, 4, 6)

    def test_in_bounds_zero_crop_area(self):
        """
        Test that crop is not ``in_bounds`` when it has zero area (undefined).
        """
        # noinspection PyArgumentList
        bb = AxisAlignedBoundingBox([1, 2], [1, 2])
        assert not crop_in_bounds(bb, 4, 6)
