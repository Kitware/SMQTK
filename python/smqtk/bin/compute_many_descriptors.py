"""
Descriptor computation helper utility. Checks data content type with respect
to the configured descriptor generator to skip content that does not match
the accepted types. Optionally, we can additionally filter out image content
whose image bytes we cannot load via ``PIL.Image.open``.
"""
import collections
import csv
import logging
import os
from typing import cast, Deque, Optional

from smqtk.algorithms import DescriptorGenerator
from smqtk.compute_functions import compute_many_descriptors
from smqtk.representation import (
    DescriptorElementFactory,
    DataSet,
    DescriptorSet,
)
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import parallel
from smqtk.utils.cli import (
    utility_main_helper,
    ProgressReporter,
    basic_cli_parser,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)
from smqtk.utils.image import is_valid_element


def default_config():
    return {
        "descriptor_generator":
            make_default_config(DescriptorGenerator.get_impls()),
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "descriptor_set":
            make_default_config(DescriptorSet.get_impls()),
        "optional_data_set":
            make_default_config(DataSet.get_impls())
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

    file_paths = [line.strip() for line in open(filelist_filepath)]

    log.info("Making descriptor factory")
    factory = DescriptorElementFactory.from_config(c['descriptor_factory'])

    log.info("Making descriptor index")
    descriptor_set = cast(
        DescriptorSet,
        from_config_dict(c['descriptor_set'],
                         DescriptorSet.get_impls())
    )

    # ``data_set`` added to within the ``iter_valid_elements`` function.
    data_set: Optional[DataSet] = None
    if c['optional_data_set']['type'] is None:
        log.info("Not saving loaded data elements to data set")
    else:
        log.info("Initializing data set to append to")
        data_set = cast(
            DataSet,
            from_config_dict(c['optional_data_set'], DataSet.get_impls())
        )

    log.info("Making descriptor generator '%s'",
             c['descriptor_generator']['type'])
    generator = cast(
        DescriptorGenerator,
        from_config_dict(c['descriptor_generator'],
                         DescriptorGenerator.get_impls())
    )

    def iter_valid_elements():
        def is_valid(file_path):
            e = DataFileElement(file_path)

            if is_valid_element(
                    e, valid_content_types=generator.valid_content_types(),
                    check_image=check_image):
                return e
            else:
                return False

        data_elements: Deque[DataFileElement] = collections.deque()
        valid_files_filter = parallel.parallel_map(is_valid,
                                                   file_paths,
                                                   name="check-file-type",
                                                   use_multiprocessing=True)
        for dfe in valid_files_filter:
            if dfe:
                yield dfe
                if data_set is not None:
                    data_elements.append(dfe)
                    if batch_size and len(data_elements) == batch_size:
                        log.debug("Adding data element batch to set (size: %d)",
                                  len(data_elements))
                        data_set.add_data(*data_elements)
                        data_elements.clear()
        # elements only collected if we have a data-set configured, so add any
        # still in the deque to the set
        if data_set is not None and data_elements:
            log.debug("Adding data elements to set (size: %d",
                      len(data_elements))
            data_set.add_data(*data_elements)

    log.info("Computing descriptors")
    m = compute_many_descriptors(iter_valid_elements(),
                                 generator,
                                 factory,
                                 descriptor_set,
                                 batch_size=batch_size,
                                 )

    # Recording computed file paths and associated file UUIDs (SHA1)
    cf = open(checkpoint_filepath, 'w')
    cf_writer = csv.writer(cf)
    try:
        pr = ProgressReporter(log.debug, 1.0).start()
        for de, descr in m:
            # We know that we are using DataFileElements going into the
            # compute_many_descriptors, so we can assume that's what comes out
            # of it as well.
            # noinspection PyProtectedMember
            cf_writer.writerow([de._filepath, descr.uuid()])
            pr.increment_report()
        pr.report()
    finally:
        del cf_writer
        cf.close()

    log.info("Done")


def cli_parser():
    parser = basic_cli_parser(__doc__)

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
    args = cli_parser().parse_args()
    config = utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    completed_files_fp = args.completed_files
    filelist_fp = args.file_list
    batch_size = args.batch_size
    check_image = args.check_image

    # Input checking
    if not filelist_fp:
        log.error("No file-list file specified")
        exit(102)
    elif not os.path.isfile(filelist_fp):
        log.error("Invalid file list path: %s", filelist_fp)
        exit(103)

    if not completed_files_fp:
        log.error("No complete files output specified")
        exit(104)

    if batch_size < 0:
        log.error("Batch size must be >= 0.")
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
