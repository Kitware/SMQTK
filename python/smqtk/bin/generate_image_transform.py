"""
Utility for transforming an input image in various standardized ways, saving
out those transformed images with standard namings. Transformations used are
configurable via a configuration file (JSON).

Configuration details:
{
    "crop": {

        "center_levels": null | int
            # If greater than 0, crop out one or more increasing smaller images
            # from a base image by cutting off increasingly larger portions of
            # the outside perimeter. Cropped image dimensions determined by the
            # dimensions of the base image and the number of crops to generate.

        "quadrant_pyramid_levels": null | int
            # If greater than 0, generate a number of crops based on a number of
            # quad-tree partitions made based on the given number of levels.
            # Partitions for all levels less than the level provides are also
            # made.

        "tile_shape": null | [width, height]
            # If not null and is a list of two integers, crop out tile windows
            # from the base image that have the width and height specified.
            # If the image width or height is not evenly divisible by the tile
            # width or height, respectively, then the crop out as many tiles as
            # neatly fit starting from the axis origin. The remaining pixels are
            # ignored.

        "tile_stride": null | [x, y]
            # If not null and is a list of two integers, crop out sub-images of
            # the above width and height (if given) with this stride. When not
            # this is not provided, the default stride is the same as the tile
            # width and height.
    },

    "brightness_levels": null | int
        # Generate a number of images with different brightness levels using
        # linear interpolation to choose levels between 0 (black) and 1
        # (original image) as well as between 1 and 2.
        # Results will not include contrast level 0, 1 or 2 images.

    "contrast_levels": null | int
        # Generate a number of images with different contrast levels using
        # linear interpolation to choose levels between 0 (black) and 1
        # (original image) as well as between 1 and 2.
        # Results will not include contrast level 0, 1 or 2 images.

}

"""

import logging
import os

import PIL.Image

import smqtk.utils.cli
import smqtk.utils.file
import smqtk.utils.parallel
from smqtk.utils.image import (
    image_crop_center_levels, image_crop_quadrant_pyramid, image_crop_tiles,
    image_brightness_intervals, image_contrast_intervals
)


def generate_image_transformations(image_path,
                                   crop_center_n, crop_quadrant_levels,
                                   crop_tile_shape, crop_tile_stride,
                                   brightness_intervals,
                                   contrast_intervals,
                                   output_dir=None,
                                   output_ext='.png'):
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
    smqtk.utils.file.safe_create_dir(output_dir)
    p_base = os.path.splitext(os.path.basename(abs_path))[0]
    p_ext = output_ext
    p_base = os.path.join(output_dir, p_base)
    image = PIL.Image.open(image_path).convert('RGB')

    def save_image(img, suffixes):
        """
        Save an image based on source image basename and an iterable of suffix
        parts that will be separated by periods.
        """
        fn = '.'.join([p_base] + list(suffixes)) + p_ext
        log.debug("Saving: %s", fn)
        img.save(fn)

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

    if crop_tile_shape and crop_tile_shape[0] > 0 and crop_tile_shape[1] > 0:
        tag = "crop_tiles"
        t_width = crop_tile_shape[0]
        t_height = crop_tile_shape[1]
        log.info("Cropping %dx%d pixel tiles from images with stride %s"
                 % (t_width, t_height, crop_tile_stride))
        # List needed to iterate generator.
        list(smqtk.utils.parallel.parallel_map(
            lambda x, y, ii:
                save_image(ii, [tag,
                                '%dx%d+%d+%d' % (t_width, t_height, x, y)]),
            image_crop_tiles(image, t_width, t_height, crop_tile_stride)
        ))

    if brightness_intervals:
        log.info("Computing brightness variants")
        for b, i in image_brightness_intervals(image, brightness_intervals):
            save_image(i, ['brightness', str(b)])

    if contrast_intervals:
        log.info("Computing contrast variants")
        for c, i in image_contrast_intervals(image, contrast_intervals):
            save_image(i, ['contrast', str(c)])


def default_config():
    return {
        "crop": {
            # 0 means disabled
            "center_levels": None,
            # 0 means disabled, 2 meaning 2x2 and 4x4
            "quadrant_pyramid_levels": None,
            # Tile shape or None for no tiling
            "tile_shape": None,
            # The stride of tiles top crop out. This defaults to the height and
            # width of the tiles to create non-overlapping chips.
            "tile_stride": None,
        },
        "brightness_levels": None,
        "contrast_levels": None,
    }


def cli_parser():
    parser = smqtk.utils.cli.basic_cli_parser(__doc__)

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
    config = smqtk.utils.cli.utility_main_helper(default_config, args)
    input_image_path = args.image
    output_dir = args.output

    if input_image_path is None:
        raise ValueError("No input image path given")

    crop_center_levels = config['crop']['center_levels']
    crop_quad_levels = config['crop']['quadrant_pyramid_levels']
    crop_tile_shape = config['crop']['tile_shape']
    crop_tile_stride = config['crop']['tile_stride']
    b_levels = config['brightness_levels']
    c_levels = config['contrast_levels']

    generate_image_transformations(
        input_image_path,
        crop_center_levels, crop_quad_levels,
        crop_tile_shape, crop_tile_stride,
        b_levels, c_levels,
        output_dir
    )


if __name__ == '__main__':
    main()
