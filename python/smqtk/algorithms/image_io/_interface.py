import abc
from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import ContentTypeValidator


class ImageReader (SmqtkAlgorithm, ContentTypeValidator):
    """
    Interface for algorithms that load a raster image matrix from a data
    element.
    """

    __slots__ = ()

    def load_as_matrix(self, data_element):
        """
        Load an image matrix from the given data element.

        Matrix return format is implementation dependant. Implementations of
        this interface should specify and describe their return type.

        Aside from a ``ValueError`` exception (documented below) the specific
        other exceptions may be raised if an image fails to load that are
        implementation dependent.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.

        :raises ValueError: The given ``data_element`` was not of a valid
            content type.

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: numpy.ndarray

        """
        self.raise_valid_element(data_element)
        # TODO: Special interaction with ImageMatrixDataElement to
        #       immediately return encapsulated matrix.
        return self._load_as_matrix(data_element)

    @abc.abstractmethod
    def _load_as_matrix(self, data_element):
        """
        Internal method to be implemented that attempts loading an image
        from the given data element and returning it as an image matrix.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: numpy.ndarray

        """


# TODO: ImageWriter interface
