import mock
import numpy as np
import pytest

from smqtk.algorithms.image_io import ImageReader
from smqtk.representation import AxisAlignedBoundingBox, DataElement


class DummyImageReader (ImageReader):
    """
    Dummy implementation of ImageReader for mocking.
    """

    @classmethod
    def is_usable(cls):
        # from Pluggable
        # Required to be True to construct a dummy instance.
        return True

    def get_config(self):
        # from Configurable
        raise NotImplementedError()

    def valid_content_types(self):
        # from ContentTypeValidator
        raise NotImplementedError()

    #
    # ImageReader abstract methods
    #

    def _load_as_matrix(self, data_element, pixel_crop=None):
        raise NotImplementedError()


def test_load_as_matrix_crop_zero_volume():
    """
    Test that a ValueError is raised when a crop bbox is passed with zero
    volume.
    """
    m_reader = mock.MagicMock(spec=ImageReader)
    m_data = mock.MagicMock(spec_set=DataElement)

    crop_bb = AxisAlignedBoundingBox([0, 0], [0, 0])
    with pytest.raises(ValueError, match=r"Volume of crop bounding box must be "
                                         r"greater than 0\."):
        ImageReader.load_as_matrix(m_reader, m_data, pixel_crop=crop_bb)


def test_load_as_matrix_crop_not_integer():
    """
    Test that a ValueError is raised when the pixel crop bbox provided does not
    report an integer type as its dtype.
    """
    m_reader = mock.MagicMock(spec=ImageReader)
    m_data = mock.MagicMock(spec_set=DataElement)

    # Create bbox with floats.
    crop_bb = AxisAlignedBoundingBox([0.0, 0.0], [1.0, 1.0])

    with pytest.raises(ValueError,
                       match=r"Crop bounding box must be composed of integer "
                             r"coordinates\. Given bounding box with dtype "
                             r".+\."):
        ImageReader.load_as_matrix(m_reader, m_data, pixel_crop=crop_bb)


def test_load_as_matrix_bad_content_type():
    """
    Test that base abstract method raises an exception when data element
    content type is a mismatch compared to reported ``valid_content_types``.

    NOTE: Uses ``DummyImageReader`` in order to pick up expected functionality
          of parent classes.
    """
    m_reader = DummyImageReader()
    m_reader.valid_content_types = mock.Mock(return_value=set())

    #: :type: DataElement
    m_e = mock.Mock(spec_set=DataElement)
    m_e.content_type.return_value = 'not/valid'

    with pytest.raises(ValueError):
        # noinspection PyCallByClass
        ImageReader.load_as_matrix(m_reader, m_e)


def test_load_as_matrix_property_shortcut():
    """
    Test that if the data-element provided has the ``matrix`` attribute, that is
    returned directly.  This is intended for use
    """
    # Mock element matrix attribute return.
    expected_mat = np.array([[1, 2, 3],
                             [6, 1, 9]])
    m_elem = mock.MagicMock(spec_set=DataElement)

    m_prop_matrix = mock.PropertyMock(return_value=expected_mat)
    type(m_elem).matrix = m_prop_matrix

    m_reader = mock.MagicMock(spec_set=ImageReader)
    actual_mat = ImageReader.load_as_matrix(m_reader, m_elem)

    np.testing.assert_allclose(actual_mat, expected_mat)

    m_prop_matrix.assert_called_once()
    m_reader._load_as_matrix.assert_not_called()


def test_load_as_matrix_success():
    """
    Test successfully passing ``load_as_matrix`` and invoking implementation
    defined ``_load_as_matrix`` method.
    """
    m_elem = mock.MagicMock(spec_set=DataElement)
    m_reader = mock.MagicMock(spec_set=ImageReader)
    crop_bb = AxisAlignedBoundingBox([0, 0], [1, 1])

    ImageReader.load_as_matrix(m_reader, m_elem, pixel_crop=crop_bb)

    m_reader._load_as_matrix.assert_called_once_with(m_elem,
                                                     pixel_crop=crop_bb)
