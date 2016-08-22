"""
Compute many descriptors from a set of file paths loaded from file.
"""
import io
import logging
import os

import PIL.Image

from smqtk.algorithms import get_descriptor_generator_impls
from smqtk.compute_functions import compute_many_descriptors
from smqtk.representation import (
    DescriptorElementFactory,
    get_descriptor_index_impls,
)
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils.bin_utils import utility_main_helper, report_progress
from smqtk.utils import plugin, parallel


def default_config():
    return {
        "descriptor_generator":
            plugin.make_config(get_descriptor_generator_impls()),
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "descriptor_index":
            plugin.make_config(get_descriptor_index_impls())
    }


def run_file_list(c, filelist_filepath, checkpoint_filepath, batch_size=None,
                  check_image=False):
    """
    Top level function handling configuration and inputs/outputs.

    :param c: Configuration dictionary (JSON)
    :type c: dict

    :param filelist_filepath: Path to a text file that lists paths to data
        files, separated by new lines.
    :type filelist_filepath: str

    :param checkpoint_filepath: Output file to which we write input filepath to
        SHA1 (UUID) relationships.
    :type checkpoint_filepath:

    :param batch_size: Optional batch size (None default) of data elements to
        process / descriptors to compute at a time. This causes files and
        stores to be written to incrementally during processing instead of
        one single batch transaction at a time.
    :type batch_size:

    :param check_image: Enable checking image loading from file before queueing
        that file for processing. If the check fails, the file is skipped
        instead of a halting exception being raised.
    :type check_image: bool

    """
    log = logging.getLogger(__name__)

    file_paths = [l.strip() for l in open(filelist_filepath)]

    log.info("Making descriptor factory")
    factory = DescriptorElementFactory.from_config(c['descriptor_factory'])

    log.info("Making descriptor index")
    #: :type: smqtk.representation.DescriptorIndex
    descriptor_index = plugin.from_plugin_config(c['descriptor_index'],
                                                 get_descriptor_index_impls())

    log.info("Making descriptor generator '%s'",
             c['descriptor_generator']['type'])
    #: :type: smqtk.algorithms.DescriptorGenerator
    generator = plugin.from_plugin_config(c['descriptor_generator'],
                                          get_descriptor_generator_impls())

    def test_image_load(dfe):
        try:
            PIL.Image.open(io.BytesIO(dfe.get_bytes()))
            return True
        except IOError, ex:
            # noinspection PyProtectedMember
            log.warn("Failed to convert '%s' bytes into an image "
                     "(error: %s). Skipping",
                     dfe._filepath, str(ex))
            return False

    def is_valid_element(fp):
        dfe = DataFileElement(fp)
        ct = dfe.content_type()
        if ct in generator.valid_content_types():
            if not check_image or test_image_load(dfe):
                return dfe
            else:
                return None
        else:
            log.debug("Skipping file (invalid content) type for "
                      "descriptor generator (fp='%s', ct=%s)",
                      str(fp), ct)
            return None

    def iter_valid_elements():
        valid_files_filter = parallel.parallel_map(is_valid_element,
                                                   file_paths,
                                                   name="check-file-type",
                                                   use_multiprocessing=True)
        for dfe in valid_files_filter:
            if dfe is not None:
                yield dfe

    log.info("Computing descriptors")
    m = compute_many_descriptors(iter_valid_elements(),
                                 generator,
                                 factory,
                                 descriptor_index,
                                 batch_size=batch_size,
                                 )

    # Recording computed file paths and associated file UUIDs (SHA1)
    cf = open(checkpoint_filepath, 'w')
    try:
        rps = [0] * 7
        for fp, descr in m:
            cf.write("{:s},{:s}\n".format(
                fp, descr.uuid()
            ))
            report_progress(log.debug, rps, 1.)
    finally:
        cf.close()

    log.info("Done")


def extend_parser(parser):
    parser.add_argument('-b', '--batch-size',
                        type=int, default=0, metavar='INT',
                        help="Number of files to batch together into a single "
                             "compute async call. This defines the "
                             "granularity of the checkpoint file in regards "
                             "to computation completed. If given 0, we do not "
                             "batch and will perform a single "
                             "``compute_async`` call on the configured "
                             "generator. Default batch size is 0.")
    parser.add_argument('--check-image',
                        default=False, action='store_true',
                        help="If se should check image pixel loading before "
                             "queueing an input image for processing. If we "
                             "cannot load the image pixels via "
                             "``PIL.Image.open``, the input image is not "
                             "queued for processing")

    # Non-config required arguments
    g_required = parser.add_argument_group("Required Arguments")
    g_required.add_argument('-f', '--file-list',
                            type=str, default=None, metavar='PATH',
                            help="Path to a file that lists data file paths. "
                                 "Paths in this file may be relative, but "
                                 "will at some point be coerced into absolute "
                                 "paths based on the current working "
                                 "directory.")
    g_required.add_argument('-p', '--completed-files',
                            default=None, metavar='PATH',
                            help='Path to a file into which we add CSV '
                                 'format lines detailing filepaths that have '
                                 'been computed from the file-list provided, '
                                 'as the UUID for that data (currently the '
                                 'SHA1 checksum of the data).')

    return parser


def main():
    description = """
    Descriptor computation helper utility. Checks data content type with respect
    to the configured descriptor generator to skip content that does not match
    the accepted types. Optionally, we can additionally filter out image content
    whose image bytes we cannot load via ``PIL.Image.open``.
    """

    args, config = utility_main_helper(default_config, description,
                                       extend_parser)
    l = logging.getLogger(__name__)

    completed_files_fp = args.completed_files
    filelist_fp = args.file_list
    batch_size = args.batch_size
    check_image = args.check_image

    # Input checking
    if not filelist_fp:
        l.error("No file-list file specified")
        exit(102)
    elif not os.path.isfile(filelist_fp):
        l.error("Invalid file list path: %s", filelist_fp)
        exit(103)

    if not completed_files_fp:
        l.error("No complete files output specified")
        exit(104)

    if batch_size < 0:
        l.error("Batch size must be >= 0.")
        exit(105)

    run_file_list(
        config,
        filelist_fp,
        completed_files_fp,
        batch_size,
        check_image
    )


if __name__ == '__main__':
    main()
