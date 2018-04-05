import io
import logging
from io import BytesIO

import PIL.Image
import numpy as np
from matplotlib import pyplot as plt

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


def overlay_saliency_map(sa_map, org_img_bytes):
    """
        overlay the saliency map on top of original image

        :param sa_map: saliency map
        :type sa_map: numpy.array

        :param org_img_bytes: Original image
        :type org_img_bytes: bytes

        :return: Overlayed image
        :rtype: bytes

        """
    sizes = np.shape(sa_map)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure(dpi=int(height))
    fig.set_size_inches((width / height), 1, forward=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(PIL.Image.open(BytesIO(org_img_bytes)))
    ax.imshow(sa_map, cmap='jet', alpha=0.5)

    fig.canvas.draw()
    np_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_data = np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    im = PIL.Image.fromarray(np_data)
    plt.close()

    b = BytesIO()
    im.save(b, format='PNG')

    return b.getvalue()
