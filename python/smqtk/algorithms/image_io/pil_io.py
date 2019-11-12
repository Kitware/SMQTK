import numpy

from six import BytesIO

from smqtk.algorithms import ImageReader
from smqtk.utils.image import crop_in_bounds

try:
    import PIL.Image
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False


class PilImageReader (ImageReader):
    """
    Image reader that uses PIL to load the image.

    This implementation may additionally raise an ``IOError`` when failing to
    to load an image.
    """

    @classmethod
    def is_usable(cls):
        return IMPORT_SUCCESS

    def __init__(self, explicit_mode=None):
        """
        Image reader that uses PIL to load the image.

        This reader returns an image matrix in [height, width] or [height,
        width, channel] dimension format, depending on source or explicit
        image mode requested.

        :param None|str explicit_mode:
            Optional explicit PIL Image mode to convert the image to before
            converting into a matrix. This should be one of the `PIL concept
            modes`_.

        :raises ValueError: The provided ``explicit_mode`` was not a valid mode
            string according to ``PIL.Image.MODES``.

        .. _PIL concept modes: https://pillow.readthedocs.io/en/3.1.x/handbook/
           concepts.html#concept-modes

        """
        super(PilImageReader, self).__init__()
        if explicit_mode and explicit_mode not in PIL.Image.MODES:
            raise ValueError("Given explicit image mode was not a valid mode "
                             "as reported by ``PIL.Image.MODES``. Given '{}'. "
                             "Available modes: {}."
                             .format(explicit_mode, PIL.Image.MODES))
        self._explicit_mode = explicit_mode

    def get_config(self):
        return {
            'explicit_mode': self._explicit_mode,
        }

    def valid_content_types(self):
        # Explicitly load standard file format drivers.
        # - confirmed idempotent from at least version 5.3.0
        # TODO: get access to lazy-loaded file format driver extensions?
        PIL.Image.preinit()
        return set(PIL.Image.MIME.values())

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
        # We may have to add a mode where we use write_temp and load from that
        # if loading large images straight from bytes-in-memory is a problem
        # and that approach actually alleviates anything.

        # Catch and raise alternate IOError exception for readability.
        try:
            #: :type: PIL.Image.Image
            img = PIL.Image.open(BytesIO(data_element.get_bytes()))
        except IOError as ex:
            ex_str = str(ex)
            if 'cannot identify image file' in ex_str:
                raise IOError("Failed to identify image from bytes provided "
                              "by {}".format(data_element))
            else:
                # pass through other exceptions
                raise

        if pixel_crop:
            if not crop_in_bounds(pixel_crop, *img.size):
                raise RuntimeError("Crop provided not within input image. "
                                   "Image shape: {}, crop: {}"
                                   .format(img.size, pixel_crop))
            img = img.crop(pixel_crop.min_vertex.tolist() +
                           pixel_crop.max_vertex.tolist())

        # If the loaded image is not already the optionally provided
        # explicit mode, convert it.
        if self._explicit_mode and img.mode != self._explicit_mode:
            img = img.convert(mode=self._explicit_mode)

        # noinspection PyTypeChecker
        return numpy.asarray(img)
