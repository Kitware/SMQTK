#!/usr/bin/env python
"""
Script for batch processing images files in the /data/roxyimages/ directory.
We use the chunk list located @
    /home/mattmann/data/exp5/image_catalog/deploy/data/archive/chunks/*/filelist_chunk_*.txt
which chunks all data files into smaller sets (~50k?).

"""

import glob
import logging
import multiprocessing
import os.path as osp
import shutil

from smqtk.utils import touch
from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.configuration import (
    DescriptorFactoryConfiguration,
    ContentDescriptorConfiguration,
)
from smqtk.representation.data_element.file_element import DataFileElement

import traceback


# Chuck files listing image files to process
CHUNK_FILES_GLOB = "/home/mattmann/data/exp5/image_catalog/deploy/data/archive/chunks/*/filelist_chunk_*.txt"
# Directory where we will detail what stages have already been completed
STAMP_FILES_DIR = "/data/local/memex/kitware/smqtk/stage_markers"
# Working directory to clean after completing every chunk for disk space conservation
CLEAN_WORK_DIR = "/data/local/memex/kitware/smqtk/work/ContentDescriptors"
# Parallel processes to run
PARALLEL = 4

# Descriptor Generator configuration type
DESCR_GENERATOR_CONFIG = "CD_CSIFT_RoxyImages_spatial"
# Descritpor Factory configuration to use 
DESCR_FACTORY_CONFIG = "LocalDiskFactory"

# Descriptor generator to use
DESCR_GENERATOR = ContentDescriptorConfiguration.new_inst(DESCR_GENERATOR_CONFIG)
DESCR_GENERATOR.PARALLEL = 1
# Descriptor Factory to use
DESCR_FACTORY = DescriptorFactoryConfiguration.new_inst(DESCR_FACTORY_CONFIG)


def check_stage(label):
    """
    Check if a stage has been completed.
    :return: True if the given stage label has been marked complete.
    """
    return osp.isfile(osp.join(STAMP_FILES_DIR, label))


def mark_stage(label):
    """
    Mark a stage identified by the given label as complete.
    """
    logging.getLogger("mark_stage").info("Marking stage '%s' complete", label)
    touch(osp.join(STAMP_FILES_DIR, label))


def process_file(file_path):
    """
    Run the configured content descriptor generator on the given file-based
    data via the configured descriptor factory.
    """
    l = logging.getLogger('process_file')
    l.info("Processing: %s", file_path)
    data = DataFileElement(file_path)
    # disregarding returned descriptor object
    try:
        DESCR_GENERATOR.compute_descriptor(data, DESCR_FACTORY)
    except Exception, ex:
        l.error("Exception occurred (%s) for image %s: %s\n"
                "%s",
                str(type(ex)), file_path, str(ex), traceback.format_exc())


def run():
    log = logging.getLogger("run")

    for chunk_file_path in glob.iglob(CHUNK_FILES_GLOB):
        chunk_file = open(chunk_file_path)
        
        stage_label = osp.basename(chunk_file_path)+'-processing'
        if not check_stage(stage_label):
            log.info("Processing stage: %s", stage_label)

            pool = multiprocessing.Pool(processes=PARALLEL)

            #for line in chunk_file:
            #    fpath = line.rstrip()
            #    log.debug("Async processing filepath: %s", fpath)
            #    pool.apply_async(process_file, args=(fpath,))

            file_paths = [line.rstrip() for line in chunk_file]
            pool.map(process_file, file_paths)

            pool.close()
            pool.join()
            del pool

            mark_stage(stage_label)
        else:
            log.info("'%s' already complete", stage_label)

        stage_label = osp.basename(chunk_file_path)+'-cleanup'
        if not check_stage(stage_label):
            log.info("Cleaning work tree for chunk '%s'", chunk_file_path)
            if osp.isdir(CLEAN_WORK_DIR):
                shutil.rmtree(CLEAN_WORK_DIR)
            mark_stage(stage_label)
        else:
            log.info("'%s' already complete", stage_label)


if __name__ == '__main__':
    initialize_logging(logging.getLogger(), logging.INFO)
    run()
