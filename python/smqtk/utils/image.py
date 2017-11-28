import io
import logging
import PIL.Image

from smqtk.representation.data_element.file_element import DataElement


def is_loadable_image(data_element):
    """
    Determine if an image is able to be loaded by PIL.

    :param data_element: A data element to check
    :type data_element: DataElement

    :return: Whether or not the image is loadable
    :rtype: bool

    Example:
    >>>

    """
    log = logging.getLogger(__name__)

    try:
        PIL.Image.open(io.BytesIO(data_element.get_bytes()))
        return True
    except IOError as ex:
        # noinspection PyProtectedMember
        log.debug("Failed to convert '%s' bytes into an image "
                  "(error: %s). Skipping", data_element, str(ex))
        return False


def is_valid_element(data_element, valid_content_types=None, check_image=False):
    """
    Determines if a given data element is valid.

    :param data_element: Data element
    :type data_element: DataElement

    :param valid_content_types: List of valid content types, or None to skip
        content type checking.
    :type valid_content_types: iterable | None

    :param check_image: Whether or not to try loading the image with PIL. This
        often catches issues that content type can't, such as corrupt images.
    :type check_image: bool

    :return: Whether or not the data element is valid
    :rtype: bool

    """
    log = logging.getLogger(__name__)

    if (valid_content_types is not None and
            data_element.content_type() not in valid_content_types):
        log.debug("Skipping file (invalid content) type for "
                  "descriptor generator (data_element='%s', ct=%s)",
                  data_element, data_element.content_type())
        return False

    if check_image and not is_loadable_image(data_element):
        return False

    return isinstance(data_element, DataElement)
