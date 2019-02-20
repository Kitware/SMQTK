import os
import unittest

from six.moves import mock
import numpy
import pytest

from smqtk.algorithms.image_io.pil_io import PilImageReader
from smqtk.representation import AxisAlignedBoundingBox
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.file_element import DataFileElement

from tests import TEST_DATA_DIR


@pytest.mark.skipif(not PilImageReader.is_usable(),
                    reason="PilImageReader is not usable in current "
                           "environment.")
class TestPilImageReader (unittest.TestCase):
    """
    Test PIL based implementation of ImageReader interface.
    """

    @classmethod
    def setUpClass(cls):
        cls.gh_image_fp = os.path.join(TEST_DATA_DIR, "grace_hopper.png")
        cls.gh_file_element = DataFileElement(cls.gh_image_fp)
        assert cls.gh_file_element.content_type() == 'image/png'

        cls.gh_cropped_image_fp = \
            os.path.join(TEST_DATA_DIR, 'grace_hopper.100x100+100+100.png')
        cls.gh_cropped_file_element = DataFileElement(cls.gh_cropped_image_fp)
        assert cls.gh_cropped_file_element.content_type() == 'image/png'
        cls.gh_cropped_bbox = AxisAlignedBoundingBox([100, 100], [200, 200])

    def test_init_invalid_mode(self):
        """
        Test that construction fails when a given ``explicit_mode``.
        """
        with pytest.raises(ValueError):
            PilImageReader(explicit_mode="not really a mode")

    def test_init_no_mode(self):
        """
        Test construction with default no-explicit image mode.
        """
        expected_mode = None
        i = PilImageReader()
        assert i._explicit_mode == expected_mode

    def test_init_valid_mode(self):
        """
        Test construction with a valid image mode.
        """
        expected_mode = "RGB"
        i = PilImageReader(explicit_mode=expected_mode)
        assert i._explicit_mode == expected_mode

    def test_configuration(self):
        """
        Test getting and constructing from configuration.
        """
        expected_exmode = 'L'
        inst1 = PilImageReader(explicit_mode=expected_exmode)
        inst1_config = inst1.get_config()

        inst2 = PilImageReader.from_config(inst1_config)
        assert inst2.get_config() == inst1_config

    def test_valid_content_types(self):
        """
        Test that the set of valid content types is not empty.
        """
        inst = PilImageReader()
        assert len(inst.valid_content_types()) > 0

    def test_load_as_matrix_no_bytes(self):
        """
        Test that a data element with no bytes fails to load.
        """
        d = DataMemoryElement(content_type='image/png')
        # Not initializing any bytes
        inst = PilImageReader()
        with pytest.raises(IOError,
                           match="Failed to identify image from bytes "
                                 "provided by DataMemoryElement"):
            inst.load_as_matrix(d)

    def test_load_as_matrix_invalid_bytes(self):
        """
        Test that data element with invalid data bytes fails to load.
        """
        d = DataMemoryElement(content_type='image/png')
        d.set_bytes("not valid bytes")

        inst = PilImageReader()
        with pytest.raises(IOError,
                           match="Failed to identify image from bytes "
                                 "provided by DataMemoryElement"):
            inst.load_as_matrix(d)

    @mock.patch('smqtk.algorithms.image_io.pil_io.PIL.Image.open')
    def test_load_as_matrix_other_exception(self, m_pil_open):
        """
        Test that some other exception raised from ``PIL.Image.open`` is
        passed through.
        """
        expected_exception = RuntimeError("Some other exception")
        m_pil_open.side_effect = expected_exception

        inst = PilImageReader()

        with pytest.raises(RuntimeError, match=str(expected_exception)):
            inst.load_as_matrix(self.gh_file_element)

    @mock.patch('smqtk.algorithms.image_io.pil_io.PIL.Image.open')
    def test_load_as_matrix_other_io_exception(self, m_pil_open):
        """
        Test that an IOError that does match conditions for alternate raise
        is raised as-is.
        """
        expected_exception = IOError("some other exception message content")
        m_pil_open.side_effect = expected_exception

        inst = PilImageReader()

        with pytest.raises(IOError, match=str(expected_exception)):
            inst.load_as_matrix(self.gh_file_element)

    def test_load_as_matrix_hopper(self):
        """
        Test loading valid data Grace Hopper image data element (native RGB
        image).
        """
        inst = PilImageReader()
        mat = inst.load_as_matrix(self.gh_file_element)
        assert isinstance(mat, numpy.ndarray)
        assert mat.dtype == numpy.uint8
        assert mat.shape == (600, 512, 3)

    def test_load_as_matrix_explicit_grayscale(self):
        """
        Test loading valid Grace Hopper image with an explicit conversion
        type to grayscale. Should result in a single channel image (only 2
        dims).
        """
        # 8-bit grayscale
        inst = PilImageReader(explicit_mode="L")
        mat = inst.load_as_matrix(self.gh_file_element)
        assert isinstance(mat, numpy.ndarray)
        assert mat.dtype == numpy.uint8
        assert mat.shape == (600, 512)

    def test_load_as_matrix_with_crop(self):
        """
        Test that passing valid crop bounding box results in the expected area.

        We load two images: the original with a crop specified, and a
        pre-cropped image. The results of each load should be the same,
        indicating the correct region from the source image is extracted.
        """
        inst = PilImageReader()
        mat_expected = inst.load_as_matrix(self.gh_cropped_file_element)
        mat_actual = inst.load_as_matrix(self.gh_file_element,
                                         pixel_crop=self.gh_cropped_bbox)
        numpy.testing.assert_allclose(mat_actual, mat_expected)

    def test_load_as_matrix_with_crop_not_integer(self):
        """
        Test passing a bounding box that is not integer aligned, which should
        raise an error in the super call.
        """
        inst = PilImageReader()
        bb = AxisAlignedBoundingBox([100, 100.6], [200, 200.2])

        with pytest.raises(ValueError, match=r"Crop bounding box must be "
                                             r"composed of integer "
                                             r"coordinates\."):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

    def test_load_as_matrix_with_crop_not_in_bounds(self):
        """
        Test that error is raised when crop bbox is not fully within the image
        bounds.
        """
        inst = PilImageReader()

        # Nowhere close
        bb = AxisAlignedBoundingBox([5000, 6000], [7000, 8000])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside left side
        bb = AxisAlignedBoundingBox([-1, 1], [2, 2])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside top side
        bb = AxisAlignedBoundingBox([1, -1], [2, 2])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside right side
        bb = AxisAlignedBoundingBox([400, 400], [513, 600])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside bottom side
        bb = AxisAlignedBoundingBox([400, 400], [512, 601])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)
