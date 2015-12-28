#!/usr/bin/env python
"""
Compute many descriptors from a set of file paths loaded from file.
"""
import cPickle
from collections import deque
import io
import json
import logging
import os

import PIL.Image

from smqtk.algorithms import get_descriptor_generator_impls
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils.bin_utils import initialize_logging, output_config
from smqtk.utils.jsmin import jsmin
from smqtk.utils.plugin import from_plugin_config, make_config


def default_config():
    return {
        "descriptor_generator":
            make_config(get_descriptor_generator_impls),
        "descriptor_factory": DescriptorElementFactory.get_default_config()
    }


def compute_many_descriptors(file_elements, descr_generator, descr_factory,
                             batch_size=100, overwrite=False,
                             procs=None, **kwds):
    """
    Compute descriptors for each data file path, yielding
    (filepath, DescriptorElement) tuple pairs in the order that they were
    input.

    :param file_elements: Iterable of DataFileElement instances of files to work
        on.
    :param descr_generator: DescriptorGenerator implementation instance
        to use to generate descriptor vectors.
    :param descr_factory: DescriptorElement factory to use when producing
        descriptor vectors.
    :param batch_size: Optional number of elements to asynchronously compute
        at a time. This is useful when it is desired for this function to yield
        results before all descriptors have been computed, yet still take
        advantage of any batch asynchronous computation optimizations a
        particular DescriptorGenerator implementation may have. If this is
        None, this function blocks until all descriptors have been generated.
    :param overwrite: If descriptors from a particular generator already exist
        for particular data, re-compute the descriptor for that data and set
        into the generated DescriptorElement.
    :param procs: Tell the DescriptorGenerator to use a specific number of
        threads/cores.
    :param kwds: Remaining keyword-arguments that are to be passed into the
        ``compute_descriptor_async`` function on the descriptor generator.

    :return: Generator that yields (filepath, DescriptorElement) for each file
        path given, in the order file paths were provided.

    """
    log = logging.getLogger(__name__)

    # Capture of generated elements in order of generation
    # - Does not use more memory as DataElements generated hang around anyway
    dfe_deque = deque()

    def data_file_element_iter():
        """
        Helper iterator to produce DataFileElement instances from file paths
        """
        for dfe in file_elements:
            dfe_deque.append(dfe)
            yield dfe

    if batch_size:
        log.debug("Computing in batches of size %d", batch_size)

        # for optimized append and popleft (O(1))
        dfe_stack = deque()
        batch_i = 0
        for dfe in data_file_element_iter():
            # checking that we can load that data as a valid image.
            try:
                PIL.Image.open(io.BytesIO(dfe.get_bytes()))
            except IOError, ex:
                log.warn("Failed to convert '%s' into an image (error: %s). "
                         "Skipping",
                         dfe._filepath, str(ex))
                continue

            dfe_stack.append(dfe)

            if len(dfe_stack) == batch_size:
                batch_i += 1
                log.debug("Computing batch %d", batch_i)
                m = descr_generator.compute_descriptor_async(
                    dfe_stack, descr_factory, overwrite, procs, **kwds
                )
                for dfe in dfe_stack:
                    yield dfe._filepath, m[dfe]
                dfe_stack.clear()

        if len(dfe_stack):
            log.debug("Computing final batch of size %d",
                      len(dfe_stack))
            m = descr_generator.compute_descriptor_async(
                dfe_stack, descr_factory, overwrite, procs, **kwds
            )
            for dfe in dfe_stack:
                yield dfe._filepath, m[dfe]
    else:
        log.debug("Using single async call")

        # Just do everything in one call
        m = descr_generator.compute_descriptor_async(
            data_file_element_iter(), descr_factory,
            overwrite, procs, **kwds
        )
        for dfe in dfe_deque:
            yield dfe._filepath, m[dfe]


def run_file_list(c, filelist_filepath, checkpoint_filepath):
    log = logging.getLogger(__name__)

    file_paths = [l.strip() for l in open(filelist_filepath)]

    log.info("Making memory factory")
    factory = DescriptorElementFactory.from_config(c['descriptor_factory'])

    log.info("Making descriptor generator '%s'",
             c['descriptor_generator']['type'])
    #: :type: smqtk.algorithms.DescriptorGenerator
    generator = from_plugin_config(c['descriptor_generator'],
                                   get_descriptor_generator_impls)
    log.info("Making descriptor generator -- Done")

    valid_file_paths = dict()
    invalid_file_paths = dict()

    def iter_valid_elements():
        for fp in file_paths:
            dfe = DataFileElement(fp)
            ct = dfe.content_type()
            if ct in generator.valid_content_types():
                valid_file_paths[fp] = ct
                yield dfe
            else:
                invalid_file_paths[fp] = ct

    log.info("Computing descriptors")
    m = compute_many_descriptors(iter_valid_elements(),
                                 generator,
                                 factory,
                                 batch_size=256,
                                 )

    # Recording computed file paths and associated file UUIDs (SHA1)
    cf = open(checkpoint_filepath, 'a')
    try:
        for fp, descr in m:
            cf.write("{:s},{:s}\n".format(
                fp, descr.uuid()
            ))
            cf.flush()
    finally:
        cf.close()

    # Output valid file and invalid file dictionaries as pickle
    log.info("Writing valid filepaths map")
    with open('valid_file_map.pickle', 'wb') as f:
        cPickle.dump(valid_file_paths, f)
    log.info("Writing invalid filepaths map")
    with open('invalid_file_map.pickle', 'wb') as f:
        cPickle.dump(invalid_file_paths, f)

    log.info("Done")


def cli_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gen-config',
                        default=None,
                        help='Optional path to output a default '
                             'JSON configuration to. This option is required '
                             'for running.')
    parser.add_argument('-d', '--debug',
                        action='store_true', default=False,
                        help='Show debug logging statements.')

    # Non-config required arguments
    g_required = parser.add_argument_group("required arguments")
    g_required.add_argument('-c', '--config',
                            type=str, default=None,
                            help="Path to the JSON configuration file.")
    g_required.add_argument('-f', '--file-list',
                            type=str, default=None,
                            help="Path to a file that lists data file paths. "
                                 "Paths in this file may be relative, but will "
                                 "at some point be coerced into absolute paths "
                                 "based on the current working directory.")
    g_required.add_argument('--completed-files',
                            default=None,
                            help='Path to a file into which we add CSV '
                                 'format lines detailing filepaths that have '
                                 'been computed from the file-list provided, '
                                 'as the UUID for that data (currently the '
                                 'SHA1 checksum of the data).')

    return parser


if __name__ == "__main__":
    p = cli_parser()
    args = p.parse_args()

    debug = args.debug
    config_fp = args.config
    out_config_fp = args.gen_config
    completed_files_fp = args.completed_files
    filelist_fp = args.file_list

    # Initialize logging
    llevel = debug and logging.DEBUG or logging.INFO
    if not logging.getLogger('smqtk').handlers:
        initialize_logging(logging.getLogger('smqtk'), llevel)
    if not logging.getLogger('__main__').handlers:
        initialize_logging(logging.getLogger('__main__'), llevel)

    l = logging.getLogger(__name__)

    # Merge loaded config with default
    config_loaded = False
    c = default_config()
    if config_fp:
        if os.path.isfile(config_fp):
            with open(config_fp) as f:
                c.update(json.loads(jsmin(f.read())))
            config_loaded = True
        else:
            l.error("Config file path not valid")
            exit(100)

    output_config(out_config_fp, c, overwrite=True)

    # Input checking
    if not config_loaded:
        l.error("No configuration provided")
        exit(101)

    if not filelist_fp:
        l.error("No file-list file specified")
        exit(102)
    elif not os.path.isfile(filelist_fp):
        l.error("Invalid file list path: %s", filelist_fp)
        exit(103)

    if not completed_files_fp:
        l.error("No complete files output specified")
        exit(104)

    run_file_list(
        c,
        filelist_fp,
        completed_files_fp,
    )
