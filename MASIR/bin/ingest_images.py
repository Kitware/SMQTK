#!/usr/bin/env python
# coding=utf-8
"""
Ingest into the data directory image files found with a given glob.

Usage:

    ingest_images.py GLOB1 [ GLOB2 ... ]

This must be called after sourcing your environment setup script else certain
utilities will not be able to be located.

"""

# bson as installed by pymongo
# noinspection PyPackageRequirements
import bson
import glob
import hashlib
import logging
import os.path as osp

from masir import IngestManager

import masir_config


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main():
    import optparse
    usage = "Usage: %prog [options] GLOB1 [ GLOB2 ... ]"
    parser = optparse.OptionParser(usage)
    parser.add_option('-d', '--data-dir',
                      help="Non-standard directory to treat as the base data "
                           "directory.")
    opts, args = parser.parse_args()

    log = logging.getLogger("main")
    data_dir = opts.data_dir or masir_config.DIR_DATA

    source_files = []
    for g in args:
        source_files.extend(glob.glob(g))

    if not source_files:
        raise ValueError("No files found with the supplied globs.")

    im = IngestManager(data_dir)
    for f in source_files:
        # Expect to possibly find some bson files in here. Skip them.
        if osp.splitext(f)[1] == '.bson':
            continue

        try:
            # if there's a found paired BSON file, pass that too
            md_filepath = osp.splitext(f)[0] + ".bson"
            if not osp.exists(md_filepath):
                md_filepath = None

            im.ingest_image(f, md_filepath)
        except IOError:
            log.warn("Not an image file: %s", f)
            continue
        except bson.InvalidBSON, ex:
            log.warn("BSON Error: %s", str(ex))
        except Exception, ex:
            log.warn("Other exception caught for file '%s':\n"
                     "    %s",
                     f, str(ex))


if __name__ == "__main__":
    main()
