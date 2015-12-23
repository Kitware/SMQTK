#!/usr/bin/env python
"""
Compute many descriptors from a set of file paths loaded from file.
"""
from collections import deque
import io
import logging
import shutil

import PIL.Image

from smqtk.representation.data_element.file_element import DataFileElement


def compute_many_descriptors(file_paths, descr_generator, descr_factory,
                             batch_size=100, overwrite=False,
                             procs=None, **kwds):
    """
    Compute descriptors for each data file path, yielding
    (filepath, DescriptorElement) tuple pairs in the order that they were
    input.

    :param file_paths: Iterable of file path strings of files to work on.
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
        for fp in file_paths:
            dfe = DataFileElement(fp)
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


def run_file_list(json_config_filepath, filelist_filepath, checkpoint_filepath):
    import json
    from smqtk.algorithms import get_descriptor_generator_impls
    from smqtk.representation import DescriptorElementFactory
    from smqtk.representation.descriptor_element.local_elements \
        import DescriptorMemoryElement
    from smqtk.utils.bin_utils import initialize_logging, logging
    from smqtk.utils.plugin import from_plugin_config

    if not logging.getLogger('smqtk').handlers:
        initialize_logging(logging.getLogger('smqtk'), logging.DEBUG)
    #if not logging.getLogger("PIL").handlers:
    #    initialize_logging(logging.getLogger("PIL"), logging.DEBUG)
    if not logging.getLogger('__main__').handlers:
        initialize_logging(logging.getLogger('__main__'), logging.DEBUG)
    log = logging.getLogger(__name__)

    file_paths = [l.strip() for l in open(filelist_filepath)]
    c = json.load(open(json_config_filepath))

    log.info("Making memory factory")
    # DEBUG
    #factory = DescriptorElementFactory.from_config(c['descriptor_factory'])
    factory = DescriptorElementFactory(DescriptorMemoryElement, {})

    log.info("Making descriptor generator '%s'",
             c['descriptor_generator']['type'])
    generator = from_plugin_config(c['descriptor_generator'],
                                   get_descriptor_generator_impls)

    valid_file_paths = dict()
    invalid_file_paths = dict()

    def iter_valid_files():
        for fp in file_paths:
            dfe = DataFileElement(fp)
            ct = dfe.content_type()
            if ct in generator.valid_content_types():
                valid_file_paths[fp] = ct
                yield fp
            else:
                invalid_file_paths[fp] = ct

    # DEBUG
    log.info("Getting first 100 valid images")
    fp_N = 100
    fp_list = []
    for fp in iter_valid_files():
        fp_list.append(fp)
        if len(fp_list) >= fp_N:
            break

    log.info("Computing descriptors")
    #m = compute_many_descriptors(iter_valid_files(),
    m = compute_many_descriptors(fp_list,
                                 generator,
                                 factory,
                                 )

    # DEBUG
    # # Recording computed file paths and associated file UUIDs (SHA1)
    # cf = open(checkpoint_filepath, 'a')
    # for fp, descr in m:
    #     cf.write("{:s},{:s}\n".format(
    #         fp, descr.uuid()
    #     ))
    # cf.close()

    # # Output valid file and invalid file dictionaries as pickle
    # import cPickle
    # log.info("Writing valid filepaths map")
    # with open('valid_file_map.pickle', 'wb') as f:
    #     cPickle.dump(valid_file_paths, f)
    # log.info("Writing invalid filepaths map")
    # with open('invalid_file_map.pickle', 'wb') as f:
    #     cPickle.dump(invalid_file_paths, f)

    # log.info("Done")

    for fp, descr in m:
        yield fp, descr        
    log.info("Computing descriptors - Done")


if __name__ == "__main__":
    run_file_list(
        "descriptor_factory.config.json",
        "files.to_compute.txt",
        "files.computed.csv",
    )
