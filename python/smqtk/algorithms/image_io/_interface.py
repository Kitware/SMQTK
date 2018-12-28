import abc
from collections import namedtuple
import numpy
from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import ContentTypeValidator


class ImageReader (SmqtkAlgorithm, ContentTypeValidator):
    """
    Interface for algorithms that load a raster image matrix from a data
    element.
    """

    __slots__ = ()

    def load_as_matrix(self, data_element, pixel_crop=None):
        """
        Load an image matrix from the given data element.

        Matrix return format is implementation dependant. Implementations of
        this interface should specify and describe their return type.

        Aside from the exceptions documented below, other exceptions may be
        raised when an image fails to load that are implementation dependent.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.
        :param None|smqtk.representation.AxisAlignedBoundingBox pixel_crop:
            Optional bounding box specifying a pixel sub-region to load from the
            given data.  If this is provided it must represent a valid
            sub-region within the loaded image, otherwise a RuntimeError is
            raised.  Handling of non-integer aligned boxes are implementation
            dependant.

        :raises RuntimeError: A crop region was specified but did not specify a
            valid sub-region of the image.
        :raises ValueError:
            This error is raised when:
                - The given ``data_element`` was not of a valid content type.
                - A ``pixel_crop`` bounding box was provided but was zero
                  volume.
                - ``pixel_crop`` bounding box vertices are not fully
                  represented by integers.

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: numpy.ndarray

        """
        self.raise_valid_element(data_element)
        if pixel_crop:
            if pixel_crop.hypervolume == 0:
                raise ValueError("Volume of crop bounding box must be greater "
                                 "than 0. Given: {}".format(pixel_crop))
            elif not issubclass(pixel_crop.dtype.type, numpy.integer):
                raise ValueError("Crop bounding box must be composed of "
                                 "integer coordinates. Given bounding box "
                                 "with dtype {}.".format(pixel_crop.dtype.type))
        # TODO: Special interaction with ImageMatrixDataElement to
        #       immediately return encapsulated matrix.
        return self._load_as_matrix(data_element, pixel_crop)

    @abc.abstractmethod
    def _load_as_matrix(self, data_element, pixel_crop=None):
        """
        Internal method to be implemented that attempts loading an image
        from the given data element and returning it as an image matrix.

        Pre-conditions:
            - ``pixel_crop`` has a non-zero volume and is composed of integer
              types.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.
        :param None|smqtk.representation.AxisAlignedBoundingBox pixel_crop:
            Optional pixel crop region to load from the given data.  If this
            is provided it must represent a valid sub-region within the loaded
            image, otherwise a RuntimeError is raised.

        :raises RuntimeError: A crop region was specified but did not specify a
            valid sub-region of the image.

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: numpy.ndarray

        """


# TODO: ImageWriter interface
