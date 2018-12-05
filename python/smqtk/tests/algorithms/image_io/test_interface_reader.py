import mock
import pytest
from smqtk.algorithms.image_io import ImageReader
from smqtk.representation import DataElement


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

    def is_loadable_image(self, data_element):
        raise NotImplementedError()

    def _load_as_matrix(self, data_element):
        raise NotImplementedError()


def test_load_as_matrix_bad_content_type():
    """
    Test that base abstract method raises an exception when data element
    content type is a mismatch compared to reported ``valid_content_types``.
    """
    m_reader = DummyImageReader()
    m_reader.valid_content_types = mock.Mock(return_value=set())

    #: :type: DataElement
    m_e = mock.Mock(spec_set=DataElement)
    m_e.content_type.return_value = 'not/valid'

    with pytest.raises(ValueError):
        # noinspection PyCallByClass
        ImageReader.load_as_matrix(m_reader, m_e)
