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
from smqtk.representation import (
    DescriptorElementFactory,
    get_descriptor_index_impls,
)
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils.bin_utils import initialize_logging, output_config
from smqtk.utils import plugin
from smqtk.utils.jsmin import jsmin


def default_config():
    return {
        "descriptor_generator":
            plugin.make_config(get_descriptor_generator_impls),
        "descriptor_factory": DescriptorElementFactory.get_default_config(),
        "descriptor_index":
            plugin.make_config(get_descriptor_index_impls)
    }


def compute_many_descriptors(file_elements, descr_generator, descr_factory,
                             descr_index, batch_size=None, overwrite=False,
                             procs=None, **kwds):
    """
    Compute descriptors for each data file path, yielding
    (filepath, DescriptorElement) tuple pairs in the order that they were
    input.

    :param file_elements: Iterable of DataFileElement instances of files to
        work on.
    :type file_elements: collections.Iterable[smqtk.representation.data_element
                                              .file_element.DataFileElement]

    :param descr_generator: DescriptorGenerator implementation instance
        to use to generate descriptor vectors.
    :type descr_generator: smqtk.algorithms.DescriptorGenerator

    :param descr_factory: DescriptorElement factory to use when producing
        descriptor vectors.
    :type descr_factory: smqtk.representation.DescriptorElementFactory

    :param descr_index: DescriptorIndex instance to add generated descriptors
        to. When given a non-zero batch size, we add descriptors to the given
        index in batches of that size. When a batch size is not given, we add
        all generated descriptors to the index after they have been generated.
    :type descr_index: smqtk.representation.DescriptorIndex

    :param batch_size: Optional number of elements to asynchronously compute
        at a time. This is useful when it is desired for this function to yield
        results before all descriptors have been computed, yet still take
        advantage of any batch asynchronous computation optimizations a
        particular DescriptorGenerator implementation may have. If this is
        None, this function blocks until all descriptors have been generated.
    :type batch_size: None | int | long

    :param overwrite: If descriptors from a particular generator already exist
        for particular data, re-compute the descriptor for that data and set
        into the generated DescriptorElement.
    :type overwrite: bool

    :param procs: Tell the DescriptorGenerator to use a specific number of
        threads/cores.
    :type procs: None | int

    :param kwds: Remaining keyword-arguments that are to be passed into the
        ``compute_descriptor_async`` function on the descriptor generator.
    :type kwds: dict

    :return: Generator that yields (filepath, DescriptorElement) for each file
        path given, in the order file paths were provided.
    :rtype: __generator[(str, smqtk.representation.DescriptorElement)]

    """
    log = logging.getLogger(__name__)

    # Capture of generated elements in order of generation
    #: :type: deque[smqtk.representation.data_element.file_element.DataFileElement]
    dfe_deque = deque()

    def data_file_element_iter():
        """
        Helper iterator to collect the file elements as we iterate over them
        when not batching.
        :rtype: __generator[smqtk.representation.data_element
                            .file_element.DataFileElement]
        """
        for fe in file_elements:
            # checking that we can load that data as a valid image.
            try:
                PIL.Image.open(io.BytesIO(fe.get_bytes()))
            except IOError, ex:
                # noinspection PyProtectedMember
                log.warn("Failed to convert '%s' into an image "
                         "(error: %s). Skipping",
                         fe._filepath, str(ex))
                continue

            dfe_deque.append(fe)

            yield fe

    if batch_size:
        log.debug("Computing in batches of size %d", batch_size)

        batch_i = 0

        for dfe in data_file_element_iter():
            # element captured in dfe_deque

            if len(dfe_deque) == batch_size:
                batch_i += 1
                log.debug("Computing batch %d", batch_i)
                m = descr_generator.compute_descriptor_async(
                    dfe_deque, descr_factory, overwrite, procs, **kwds
                )

                log.debug("-- adding to index")
                descr_index.add_many_descriptors(m.itervalues())

                log.debug("-- yielding generated descriptor elements")
                for e in dfe_deque:
                    # noinspection PyProtectedMember
                    yield e._filepath, m[e]

                dfe_deque.clear()

        if len(dfe_deque):
            log.debug("Computing final batch of size %d",
                      len(dfe_deque))
            m = descr_generator.compute_descriptor_async(
                dfe_deque, descr_factory, overwrite, procs, **kwds
            )

            log.debug("-- adding to index")
            descr_index.add_many_descriptors(m.values())

            log.debug("-- yielding generated descriptor elements")
            for dfe in dfe_deque:
                # noinspection PyProtectedMember
                yield dfe._filepath, m[dfe]

    else:
        log.debug("Using single async call")

        # Just do everything in one call
        log.debug("Computing descriptors")
        m = descr_generator.compute_descriptor_async(
            data_file_element_iter(), descr_factory,
            overwrite, procs, **kwds
        )

        log.debug("Adding to index")
        descr_index.add_many_descriptors(m.itervalues())

        log.debug("yielding generated elements")
        for dfe in dfe_deque:
            # noinspection PyProtectedMember
            yield dfe._filepath, m[dfe]


def run_file_list(c, filelist_filepath, checkpoint_filepath, batch_size=None):
    """
    Top level function handling configuration and inputs/outputs.

    :param c: Configuration dictionary (JSON)
    :type c: dict

    :param filelist_filepath: Path to a text file that lists paths to image
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

    """
    log = logging.getLogger(__name__)

    file_paths = [l.strip() for l in open(filelist_filepath)]

    log.info("Making descriptor factory")
    factory = DescriptorElementFactory.from_config(c['descriptor_factory'])

    log.info("Making descriptor index")
    #: :type: smqtk.representation.DescriptorIndex
    descriptor_index = plugin.from_plugin_config(c['descriptor_index'],
                                                 get_descriptor_index_impls)

    log.info("Making descriptor generator '%s'",
             c['descriptor_generator']['type'])
    #: :type: smqtk.algorithms.DescriptorGenerator
    generator = plugin.from_plugin_config(c['descriptor_generator'],
                                          get_descriptor_generator_impls)

    valid_file_paths = dict()
    invalid_file_paths = dict()

    def iter_valid_elements():
        """
        :rtype:
            __generator[smqtk.representation.data_element
                        .file_element.DataFileElement]
        """
        for p in file_paths:
            dfe = DataFileElement(p)
            ct = dfe.content_type()
            if ct in generator.valid_content_types():
                yield dfe
            else:
                log.debug("Skipping file (invalid content) type for "
                          "descriptor generator (fp='%s', ct=%s)",
                          p, ct)

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
        for fp, descr in m:
            cf.write("{:s},{:s}\n".format(
                fp, descr.uuid()
            ))
    finally:
        cf.close()

    # Output valid file and invalid file dictionaries as pickle
    log.info("Writing valid filepaths map")
    with open('file_map.valid.pickle', 'wb') as f:
        cPickle.dump(valid_file_paths, f, -1)
    log.info("Writing invalid filepaths map")
    with open('file_map.invalid.pickle', 'wb') as f:
        cPickle.dump(invalid_file_paths, f, -1)

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
    parser.add_argument('-b', '--batch-size',
                        type=int, default=256,
                        help="Number of files to batch together into a single "
                             "compute async call. This defines the "
                             "granularity of the checkpoint file in regards "
                             "to computation completed. If given 0, we do not "
                             "batch and will perform a single "
                             "``compute_async`` call on the configured "
                             "generator. Default batch size is 256.")

    # Non-config required arguments
    g_required = parser.add_argument_group("required arguments")
    g_required.add_argument('-c', '--config',
                            type=str, default=None,
                            help="Path to the JSON configuration file.")
    g_required.add_argument('-f', '--file-list',
                            type=str, default=None,
                            help="Path to a file that lists data file paths. "
                                 "Paths in this file may be relative, but "
                                 "will at some point be coerced into absolute "
                                 "paths based on the current working "
                                 "directory.")
    g_required.add_argument('--completed-files',
                            default=None,
                            help='Path to a file into which we add CSV '
                                 'format lines detailing filepaths that have '
                                 'been computed from the file-list provided, '
                                 'as the UUID for that data (currently the '
                                 'SHA1 checksum of the data).')

    return parser


def main():
    p = cli_parser()
    args = p.parse_args()

    debug = args.debug
    config_fp = args.config
    out_config_fp = args.gen_config
    completed_files_fp = args.completed_files
    filelist_fp = args.file_list
    batch_size = args.batch_size

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

    if batch_size < 0:
        l.error("Batch size must be >= 0.")
        exit(105)

    run_file_list(
        c,
        filelist_fp,
        completed_files_fp,
        batch_size,
    )


if __name__ == '__main__':
    main()
