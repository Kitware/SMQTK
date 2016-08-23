"""
Utility for transforming an input image in various standardized ways, saving
out those transformed images with standard namings. Transformations used are
configurable via a configuration file (JSON).
"""

import logging
import os

import PIL.Image
import PIL.ImageEnhance
import numpy

import smqtk.utils.bin_utils
import smqtk.utils.parallel


__author__ = "paul.tunison@kitware.com"


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
    for i in xrange(1, n_crops + 1):
        # crop wants: [xmin, ymin, xmax, ymax]
        t = zip(x_points[[i, -i - 1]], y_points[[i, -i - 1]])
        yield i, image.crop(t[0] + t[1])


def image_crop_quadrant_pyramid(image, n_levels):
    """
    Generate a number of crops based on a number of quadrant sub-partitions made
    based on the given number of levels.

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

    for l in xrange(1, n_levels + 1):
        l_sq = 2**l
        xs = numpy.linspace(0, image.width, l_sq + 1, endpoint=True, dtype=int)
        ys = numpy.linspace(0, image.height, l_sq + 1, endpoint=True, dtype=int)
        for j in xrange(l_sq):
            for i in xrange(l_sq):
                yield (
                    l,
                    (i, j),
                    image.crop([xs[i], ys[j], xs[i + 1], ys[j + 1]])
                )


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


def generate_image_transformations(image_path,
                                   crop_center_n, crop_quadrant_levels,
                                   brigntness_intervals,
                                   contrast_intervals,
                                   output_dir=None):
    """
    Transform an input image into different crops or other transforms,
    outputting results to the given output directory without overwriting or
    otherwise changing the input image.

    By default, if not told otherwise, we will write output images in the same
    directory as the source image. Output images share a core filename as the
    source image, but with extra suffix syntax to differentiate produced images
    from the original. Output images will share the same image extension as the
    source image.
    """
    log = logging.getLogger(__name__)

    abs_path = os.path.abspath(image_path)
    output_dir = output_dir or os.path.dirname(abs_path)
    p_base, p_ext = os.path.splitext(os.path.basename(abs_path))
    p_base = os.path.join(output_dir, p_base)
    image = PIL.Image.open(image_path)

    def save_image(i, suffixes):
        fn = '.'.join([p_base] + list(suffixes)) + p_ext
        log.debug("Saving: %s", fn)
        i.save(fn)

    if crop_center_n:
        log.info("Computing center crops")
        tag = "crop_centers"
        for l, c in image_crop_center_levels(image, crop_center_n):
            save_image(c, [tag, str(l)])

    if crop_quadrant_levels:
        log.info("Computing quadrant crops")
        tag = "crop_quadrants"
        for l, (i, j), c in image_crop_quadrant_pyramid(image,
                                                        crop_quadrant_levels):
            save_image(c, [tag, str(l), "q_{:d}_{:d}".format(i, j)])

    if brigntness_intervals:
        log.info("Computing brightness variants")
        for b, i in image_brightness_intervals(image, brigntness_intervals):
            save_image(i, ['brightness', str(b)])

    if contrast_intervals:
        log.info("Computing contrast variants")
        for c, i in image_contrast_intervals(image, contrast_intervals):
            save_image(i, ['contrast', str(c)])


def default_config():
    return {
        "crop": {
            # 0 means disabled
            "center_levels": 3,
            # 0 means disabled, 2 meaning 2x2 and 4x4
            "quadrant_pyramid_levels": 2,
        },
        "brightness_levels": 3,
        "contrast_levels": 3,
    }


def cli_parser():
    parser = smqtk.utils.bin_utils.basic_cli_parser(__doc__)

    g_io = parser.add_argument_group("Input/Output")
    g_io.add_argument("-i", "--image",
                      help="Image to produce transformations for.")
    g_io.add_argument("-o", "--output",
                      help="Directory to output generated images to. By "
                           "default, if not told otherwise, we will write "
                           "output images in the same directory as the source "
                           "image. Output images share a core filename as the "
                           "source image, but with extra suffix syntax to "
                           "differentiate produced images from the original. "
                           "Output images will share the same image extension "
                           "as the source image.")
    return parser


def main():
    args = cli_parser().parse_args()
    config = smqtk.utils.bin_utils.utility_main_helper(default_config, args)
    input_image_path = args.image
    output_dir = args.output

    if input_image_path is None:
        raise ValueError("No input image path given")

    crop_center_levels = config['crop']['center_levels']
    crop_quad_levels = config['crop']['quadrant_pyramid_levels']
    b_levels = config['brightness_levels']
    c_levels = config['contrast_levels']

    generate_image_transformations(
        input_image_path,
        crop_center_levels, crop_quad_levels,
        b_levels, c_levels,
        output_dir
    )


if __name__ == '__main__':
    main()
