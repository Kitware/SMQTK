import io
import logging
import PIL.Image

from smqtk.representation.data_element.file_element import DataFileElement


def is_loadable_image(dfe):
    """
    Determine if an image is able to be loaded by PIL.

    :param dfe: A data file element to check
    :type dfe: DataFileElement

    :return: Whether or not the image is loadable
    :rtype: bool

    """
    log = logging.getLogger(__name__)

    try:
        PIL.Image.open(io.BytesIO(dfe.get_bytes()))
        return True
    except IOError, ex:
        # noinspection PyProtectedMember
        log.warn("Failed to convert '%s' bytes into an image "
                 "(error: %s). Skipping", dfe._filepath, str(ex))
        return False


def is_valid_element(fp, valid_content_types=None, check_image=False):
    """
    Determines if a given filepath is a valid data file element.

    :param fp: File path to element
    :type fp: str

    :param valid_content_types: List of valid content types, or None to skip
        content type checking.
    :type valid_content_types: iterable | None

    :param check_image: Whether or not to try loading the image with PIL. This
        often catches issues that content type can't, such as corrupt images.
    :type check_image: bool

    :return: The data file element in the event of a valid element, or None if
        it's invalid.
    :rtype: DataFileElement | None

    """
    log = logging.getLogger(__name__)
    dfe = DataFileElement(fp)

    if (valid_content_types is not None and
        dfe.content_type() not in valid_content_types):
        log.debug("Skipping file (invalid content) type for "
                  "descriptor generator (fp='%s', ct=%s)",
                  str(fp), dfe.content_type())
        return None

    if check_image and not is_loadable_image(dfe):
        return None

    return dfe
