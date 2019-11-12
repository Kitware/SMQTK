from six.moves import mock
import numpy as np
import pytest

from smqtk.algorithms.image_io import ImageReader
from smqtk.representation import AxisAlignedBoundingBox, DataElement
from smqtk.representation.data_element.matrix import MatrixDataElement


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


def test_is_valid_element_no_matrix():
    """
    Test handling the DataElement instances without a matrix property (i.e.
    "normal" way).
    """
    test_de = mock.MagicMock(spec=DataElement)

    m_inst = DummyImageReader()
    m_inst.valid_content_types = mock.MagicMock(return_value={'a', 'b'})

    test_de.content_type.return_value = 'a'
    assert ImageReader.is_valid_element(m_inst, test_de)

    test_de.content_type.return_value = 'b'
    assert ImageReader.is_valid_element(m_inst, test_de)

    test_de.content_type.return_value = 'c'
    assert not ImageReader.is_valid_element(m_inst, test_de)

    # Descended into ContentTypeValidator super-method, which should have
    # requested DE content type
    assert test_de.content_type.call_count == 3
    assert m_inst.valid_content_types.call_count == 3


def test_is_valid_element_has_matrix():
    """
    Test handling a DataElement instance that does have a ``matrix`` property
    """
    test_de = mock.MagicMock(spec=MatrixDataElement)
    m_inst = mock.MagicMock(spec=ImageReader)

    # Matrix return value should be appropriate type: None | ndarray
    test_de.matrix = None
    assert ImageReader.is_valid_element(m_inst, test_de)

    test_de.matrix = np.eye(3)
    assert ImageReader.is_valid_element(m_inst, test_de)


def test_is_valid_element_has_matrix_invalid_value_type():
    """
    Test that exception is raise if found matrix attribute value is of an
    unexpected type.
    """
    test_de = mock.MagicMock(spec=MatrixDataElement)
    m_inst = DummyImageReader()

    test_de.matrix = "something not a matrix"
    with pytest.raises(AssertionError,
                       match="Element `matrix` property return should either "
                             "be a matrix or None."):
        ImageReader.is_valid_element(m_inst, test_de)


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

    m_reader = mock.MagicMock(spec=ImageReader)
    m_reader._get_matrix_property = \
        mock.MagicMock(wraps=ImageReader._get_matrix_property)
    m_reader._load_as_matrix = mock.MagicMock()
    actual_mat = ImageReader.load_as_matrix(m_reader, m_elem)

    np.testing.assert_allclose(actual_mat, expected_mat)

    m_prop_matrix.assert_called_once()
    m_reader._load_as_matrix.assert_not_called()


def test_load_as_matrix_success():
    """
    Test successfully passing ``load_as_matrix`` and invoking implementation
    defined ``_load_as_matrix`` method (no ``matrix`` property on data elem).
    """
    m_elem = mock.MagicMock(spec_set=DataElement)
    crop_bb = AxisAlignedBoundingBox([0, 0], [1, 1])

    m_reader = mock.MagicMock(spec=ImageReader)
    m_reader._get_matrix_property = \
        mock.MagicMock(wraps=ImageReader._get_matrix_property)
    m_reader._load_as_matrix = mock.MagicMock()
    ImageReader.load_as_matrix(m_reader, m_elem, pixel_crop=crop_bb)

    m_reader._load_as_matrix.assert_called_once_with(m_elem,
                                                     pixel_crop=crop_bb)
