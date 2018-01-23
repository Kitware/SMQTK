#!/usr/bin/env python
"""
Transform DescriptorFileElement instances into Postgres version
"""

import cPickle
import logging
import multiprocessing
import os
import re

from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import DescriptorFileElement
from smqtk.representation.descriptor_element.postgres import PostgresDescriptorElement
from smqtk.utils import bin_utils
from smqtk.utils import file_utils


ROOT_DIR = "/data/kitware/smqtk/image_cache_cnn_compute/descriptors"


file_element_config = {
    'save_dir': ROOT_DIR,
    'subdir_split': 10,
}


psql_element_config = {
    'db_name': 'smqtk',
    'db_host': 'localhost',
    'db_port': 6432,  # PgBouncer port
    'db_user': 'smqtk',
    'db_pass': 'some-password',
}


file_element_factory = DescriptorElementFactory(
    DescriptorFileElement,
    file_element_config,
)


psql_element_factory = DescriptorElementFactory(
    PostgresDescriptorElement,
    psql_element_config,
)


fname_re = re.compile('(\w+)\.(\w+)\.vector\.npy')


def transfer_vector(type_str, uuid_str):
    pd = psql_element_factory(type_str, uuid_str)
    if not pd.has_vector():
        fd = file_element_factory(type_str, uuid_str)
        # removing the "-0" artifacts
        pd.set_vector( fd.vector() + 0 )


def proc_transfer(in_queue):
    running = True
    while running:
        packet = in_queue.get()
        if packet:
            type_str, uuid_str = packet
            transfer_vector(type_str, uuid_str)
        else:
            running = False


def main():
    bin_utils.initialize_logging(logging.getLogger(), logging.DEBUG)
    log = logging.getLogger(__name__)

    # For each file in descriptor vector file tree, load from file
    # [type, uuid, vector] and insert into PSQL element.

    log.info("Setting up parallel environment")
    in_queue = multiprocessing.Queue()
    workers = []
    for i in xrange(multiprocessing.cpu_count()):
        p = multiprocessing.Process(
            target=proc_transfer,
            args=(in_queue,)
        )
        workers.append(p)
        p.start()

    try:
        log.info("Loading filename list")
        with open("descriptor_file_names.5.3mil.pickle") as f:
            fname_list = cPickle.load(f)

        log.info("Running through filename list")
        for n in fname_list:
            m = fname_re.match(n)
            assert m

            type_str = m.group(1)
            uuid_str = m.group(2)

            #print type_str, uuid_str
            #break
            in_queue.put( (type_str, uuid_str) )

        log.info("Sending worker terminal packets")
        for w in workers:
            in_queue.put(None)

    except:
        log.info("Terminating workers")
        for w in workers:
            w.terminate()

    finally:
        log.info("Waiting for workers to complete")
        for w in workers:
            w.join()
        log.info("Workers joined")


if __name__ == '__main__':
    main()

