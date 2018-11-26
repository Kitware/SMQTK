import numpy

from six import BytesIO

from smqtk.algorithms import ImageReader

try:
    import PIL.Image
    IMPORT_SUCCESS = True
except ImageReader:
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

    def _load_as_matrix(self, data_element):
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

        # If the loaded image is not already the optionally provided
        # explicit mode, convert it.
        if self._explicit_mode and img.mode != self._explicit_mode:
            img = img.convert(mode=self._explicit_mode)

        # noinspection PyTypeChecker
        return numpy.asarray(img)
