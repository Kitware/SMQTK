#!/usr/bin/env python
"""
Create an ingest of files in a specified directory.
"""

import glob
import json
import logging
import os.path as osp

from smqtk.data_rep import get_data_set_impls
from smqtk.data_rep.data_element_impl.file_element import DataFileElement
from smqtk.utils import bin_utils, plugin


def default_config():
    return {
        "data_set": plugin.make_config(get_data_set_impls)
    }


def main():
    usage = "%prog [options] GLOB [ GLOB [ ... ] ]"
    description = "Add a set of local system files to a data set via " \
                  "explicit paths or shell-style glob strings."

    parser = bin_utils.SMQTKOptParser(usage, description=description)
    parser.add_option('-c', '--config',
                      help="Path to the JSON configuration file")
    parser.add_option('--output-config',
                      help="Optional path to output a default configuration "
                           "file to. This output file should be modified and "
                           "used for this executable.")
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                      help='Add debug messaged to output logging.')
    opts, args = parser.parse_args()

    bin_utils.initialize_logging(logging.getLogger(),
                                 logging.INFO - (10*opts.verbose))
    log = logging.getLogger("main")

    # output configuration dictionary when asked for.
    bin_utils.output_config(opts.output_config, default_config(), log)

    with open(opts.config, 'r') as f:
        config = json.load(f)

    #: :type: smqtk.data_rep.DataSet
    ds = plugin.from_plugin_config(config['data_set'], get_data_set_impls)
    log.debug("Script arguments:\n%s" % args)

    def ingest_file(fp):
        ds.add_data(DataFileElement(fp))

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
