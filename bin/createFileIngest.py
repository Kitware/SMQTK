#!/usr/bin/env python
"""
Create an ingest of files in a specified directory.
"""

import glob
import logging
import os.path as osp

from smqtk.data_rep.data_element_impl.file_element import FileElement
from smqtk.utils import bin_utils
from smqtk.utils.configuration import DataSetConfiguration
import smqtk_config


def main():
    usage = "%prog [options] GLOB [ GLOB [ ... ] ]"
    description = "Create a file-based ingest from a set of local file paths " \
                  "or shell-style glob strings."

    parser = bin_utils.SMQTKOptParser(usage, description=description)
    parser.add_option('-s', '--set-label',
                      help="Configured ingest to 'ingest' into.")
    parser.add_option('-l', '--list-ingests', action='store_true',
                      default=False,
                      help="List available ingests we can ingest new data "
                           "into. See the system_config.json file in the etc "
                           "directory for more details.")
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                      help='Add debug messaged to output logging.')
    opts, args = parser.parse_args()

    bin_utils.initializeLogging(logging.getLogger(),
                                logging.INFO - (10*opts.verbose))
    log = logging.getLogger("main")

    if opts.list_ingests:
        # Find labels for configured data sets that are of the FileSet type
        file_ds_labels = [
            l
            for l, dsc in smqtk_config.SYSTEM_CONFIG['DataSets'].iteritems()
            if dsc['type'] == "FileSet"
        ]

        log.info("")
        log.info("Available File-based datasets:")
        for k in sorted(file_ds_labels):
            log.info("\t%s", k)
        log.info("")
        exit(0)

    if opts.set_label is None:
        log.info("")
        log.info("ERROR: Please provide data set configuration label.")
        log.info("")
        exit(1)

    fds = DataSetConfiguration.new_inst(opts.set_label)
    log.debug("Script arguments:\n%s" % args)

    def ingest_file(fp):
        fds.add_data(FileElement(fp))

    for f in args:
        f = osp.expanduser(f)
        if osp.isfile(f):
            ingest_file(f)
        else:
            log.debug("Expanding glob: %s" % f)
            for g in glob.glob(f):
                ingest_file(g)


if __name__ == '__main__':
    main()
