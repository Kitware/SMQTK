import os
import unittest

import mock
import numpy
import pytest

from smqtk.algorithms.image_io.pil_io import PilImageReader
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR


FP_IMAGE_GRACE_HOPPER = os.path.join(TEST_DATA_DIR, "grace_hopper.png")


@pytest.mark.skipif(not PilImageReader.is_usable(),
                    reason="PilImageReader is not usable in current "
                           "environment.")
class TestPilImageReader (unittest.TestCase):
    """
    Test PIL based implementation of ImageReader interface.
    """

    @classmethod
    def setUpClass(cls):
        cls.gh_file_element = DataFileElement(FP_IMAGE_GRACE_HOPPER)
        assert cls.gh_file_element.content_type() == 'image/png'

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

    def test_get_matrix_no_bytes(self):
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

    def test_get_matrix_invalid_bytes(self):
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
    def test_get_matrix_other_exception(self, m_pil_open):
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
    def test_get_matrix_other_io_exception(self, m_pil_open):
        """
        Test that an IOError that does match conditions for alternate raise
        is raised as-is.
        """
        expected_exception = IOError("some other exception message content")
        m_pil_open.side_effect = expected_exception

        inst = PilImageReader()

        with pytest.raises(IOError, match=str(expected_exception)):
            inst.load_as_matrix(self.gh_file_element)

    def test_get_matrix_hopper(self):
        """
        Test loading valid data Grace Hopper image data element (native RGB
        image).
        """
        inst = PilImageReader()
        mat = inst.load_as_matrix(self.gh_file_element)
        assert isinstance(mat, numpy.ndarray)
        assert mat.dtype == numpy.uint8
        assert mat.shape == (600, 512, 3)

    def test_get_matrix_explicit_grayscale(self):
        """
        Test loading valid Grace Hopper image with an explicit conversion
        type to grayscale.
        """
        # 8-bit grayscale
        inst = PilImageReader(explicit_mode="L")
        mat = inst.load_as_matrix(self.gh_file_element)
        assert isinstance(mat, numpy.ndarray)
        assert mat.dtype == numpy.uint8
        assert mat.shape == (600, 512)
