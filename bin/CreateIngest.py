#!/usr/bin/env python
"""
Create an ingest of files in a specified directory.
"""

import glob
import logging
import os.path as osp

from SMQTK.utils.configuration import IngestConfiguration


def main():
    import optparse

    usage = "%prog [options] GLOB [ GLOB [ ... ] ]"
    description = "Create a formal ingest from a set of file paths or " \
                  "shell-style glob strings."

    parser = optparse.OptionParser(usage, description=description)
    parser.add_option('-i', '--ingest',
                      help="Configured ingest to 'ingest' into.")
    parser.add_option('-l', '--list-ingests', action='store_true',
                      default=False,
                      help="List available ingests we can ingest new data "
                           "into. See the system_config.json file in the etc "
                           "directory for more details.")
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                      help='Add debug messaged to output logging.')
    opts, args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    if opts.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if opts.list_ingests:
        print
        print "Available Ingests:"
        for k in sorted(IngestConfiguration.available_ingest_labels()):
            print "\t%s" % k
        print
        exit(0)

    if opts.ingest is None:
        print
        print "ERROR: Please provide an ingest label."
        print
        exit(1)

    ingest_config = IngestConfiguration(opts.ingest)
    ingest = ingest_config.get_ingest_instance()
    print "Script arguments:\n%s" % args
    for g in args:
        g = osp.expanduser(g)
        if osp.isfile(g):
            ingest.add_data_file(g)
        else:
            print "Expanding glob: %s" % g
            for fp in glob.glob(g):
                ingest.add_data_file(fp)


if __name__ == '__main__':
    main()
