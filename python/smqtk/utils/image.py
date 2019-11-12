import io
import logging
import numpy
import PIL.Image
import PIL.ImageEnhance

from six.moves import range

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


def is_valid_element(data_element, valid_content_types=None,
                     check_image=False):
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


def image_crop_center_levels(image, n_crops):
    """
    Crop out one or more increasing smaller images from a base image by cutting
    off increasingly larger portions of the outside perimeter. Cropped image
    dimensions determined by the dimensions of the base image and the number of
    crops to generate.

    :param image: The base image array. This is not modified,.
    :type image: PIL.Image.Image

    :param n_crops: Number of crops to generate.
    :type n_crops: int

    :return: Generator yielding paired level number and PIL.Image.Image tuples.
        Cropped images have not loaded/copied yet, so changes to the original
        image will affect them.
    :rtype: __generator[(int, PIL.Image.Image)]

    """
    n_crops = int(n_crops)
    if n_crops <= 0:
        raise ValueError("Can't produce 0 or negative crops")

    #: :type: numpy.ndarray
    x_points = numpy.linspace(0, image.width, 2 + n_crops * 2, dtype=int)
    #: :type: numpy.ndarray
    y_points = numpy.linspace(0, image.height, 2 + n_crops * 2, dtype=int)

    # Outside edges of generated points in the original image size
    for i in range(1, n_crops + 1):
        # crop wants: [xmin, ymin, xmax, ymax]
        t = zip(x_points[[i, -i - 1]], y_points[[i, -i - 1]])
        yield i, image.crop(t[0] + t[1])


def image_crop_quadrant_pyramid(image, n_levels):
    """
    Generate a number of crops based on a number of quadrant sub-partitions
    made based on the given number of levels.

    For example, 1 level would yield 4 crops from the 2x2 partition of the
    image. 2 levels would yield 20 crops for the 2x2 partition and the 4x4
    partition. 3 levels would yield the partitions of 2x2, 4x4 and 8x8 yielding
    84 crops, etc. General rule: (2^i)^2 for i in [1 n_levels]

    :param image: Image to crop in quadrant partitions
    :type image: PIL.Image.Image

    :param n_levels: Number of quadrant levels to generate crops for.
    :type n_levels: int

    :return: Generator yielding paired level, quadrant position and
        PIL.Image.Image tuples. Quadrant position is in (x, y) format.
        Crop images have not loaded/copied yet, so changes to the original
        image will affect them.
    :rtype: __generator[(int, (int, int), PIL.Image.Image)]

    """
    n_crops = int(n_levels)
    if n_crops <= 0:
        raise ValueError("Can't produce 0 or negative levels")

    for l in range(1, n_levels + 1):
        l_sq = 2**l
        xs = numpy.linspace(0, image.width, l_sq + 1,
                            endpoint=True, dtype=int)
        ys = numpy.linspace(0, image.height, l_sq + 1,
                            endpoint=True, dtype=int)
        for j in range(l_sq):
            for i in range(l_sq):
                yield (
                    l,
                    (i, j),
                    image.crop([xs[i], ys[j], xs[i + 1], ys[j + 1]])
                )


def image_crop_tiles(image, tile_width, tile_height, stride=None):
    """
    Crop out tile windows from the base image that have the width and height
    specified.

    If the image width or height is not evenly divisible by the tile width or
    height, respectively, then the crop out as many tiles as neatly fit
    starting from the axis origin. The remaining pixels are ignored.

    :param image: Image to crop tiles from.
    :type image: PIL.Image.Image

    :param tile_width: Tile crop width in pixels.
    :type tile_width: int

    :param tile_height: Tile crop height in pixels.
    :type tile_height: int

    :param stride: Optional tuple of integer pixel stride for cropping out sub-
        images. When this is None, the stride is the same as the width and
        height of the requested sub-images.
    :type stride: None | (int, int)

    :return: Generator yielding tuples containing a cropped image and its
        upper-left xy position in the original image.
    :rtype: __generator[(int, int, PIL.Image.Image)]

    """
    if stride:
        stride_x, stride_y = map(int, stride)
    else:
        stride_x = tile_width
        stride_y = tile_height

    # upper-left xy pixel coordinates for sub-images.
    y = 0
    while (y + tile_height) < image.height:
        x = 0
        while (x + tile_width) < image.width:
            t = image.crop([x, y, x+tile_width, y+tile_height])
            yield (x, y, t)
            x += stride_x
        y += stride_y


def image_brightness_intervals(image, n):
    """
    Generate a number of images with different brightness levels using linear
    interpolation to choose levels between 0 (black) and 1 (original image) as
    well as between 1 and 2.

    Results will not include contrast level 0, 1 or 2 images.

    """
    n = int(n)
    if n <= 0:
        raise ValueError("Can't produce 0 intervals")

    b = numpy.linspace(0, 1, n+2, endpoint=True, dtype=float)
    for v in b[1:-1]:
        yield v, PIL.ImageEnhance.Brightness(image).enhance(v)
    b = numpy.linspace(1, 2, n + 2, endpoint=True, dtype=float)
    for v in b[1:-1]:
        yield v, PIL.ImageEnhance.Brightness(image).enhance(v)


def image_contrast_intervals(image, n):
    """
    Generate a number of images with different contrast levels using linear
    interpolation to choose levels between 0 (black) and 1 (original image) as
    well as between 1 and 2.

    Results will not include contrast level 0, 1 or 2 images.

    """
    n = int(n)
    if n <= 0:
        raise ValueError("Can't produce 0 intervals")

    b = numpy.linspace(0, 1, n + 2, endpoint=True, dtype=float)
    for v in b[1:-1]:
        yield v, PIL.ImageEnhance.Contrast(image).enhance(v)
    b = numpy.linspace(1, 2, n + 2, endpoint=True, dtype=float)
    for v in b[1:-1]:
        yield v, PIL.ImageEnhance.Contrast(image).enhance(v)


def crop_in_bounds(bbox, im_width, im_height):
    """
    Check if this crop specification is within a given parent bounds
    specification.

    Thus function does NOT care if the input bounding box is integer aligned.

    :param smqtk.representation.AxisAlignedBoundingBox bbox:
        Bounding box representing a sub-region within an image. This must be a
        2 dimensional bounding box.
    :param int im_width:
        Parent image full width in pixels.
    :param int im_height:
        Parent image full height in pixels.

    :return: If this crop specification lies fully within the given
        bounds.  Touching the edge counts as being "fully within".
    :rtype: bool
    """
    bbox_dim = bbox.ndim
    assert bbox_dim == 2, \
        "``crop_in_bounds`` requires an image-space bounding box (2 " \
        "dimensional), given bounding box with dimension {}." \
        .format(bbox_dim)

    log = logging.getLogger('.'.join((__package__, 'crop_in_bounds')))

    ul_x, ul_y = bbox.min_vertex
    lr_x, lr_y = bbox.max_vertex

    in_bounds = True
    if not ((0 <= ul_x <= im_width) and (0 <= ul_y <= im_height)):
        log.warning("Upper-left coordinate outside image bounds ([w,h] "
                    "image dimensions: {}, given upper-left: {})"
                    .format((im_width, im_height), (ul_x, ul_y)))
        in_bounds = False
    if not ((0 <= lr_x <= im_width) and (0 <= lr_y <= im_height)):
        log.warning("Lower-right coordinate outside image bounds "
                    "([w, h] image dimensions: {}, given "
                    "lower-right: {})"
                    .format((im_width, im_height), (lr_x, lr_y)))
        in_bounds = False
    if not (((lr_x - ul_x) > 0) and ((lr_y - ul_y) > 0)):
        log.warning("Pixel crop region area is zero (ul: {}, lr: {})."
                    .format((ul_x, ul_y), (lr_x, lr_y)))
        in_bounds = False
    return in_bounds
