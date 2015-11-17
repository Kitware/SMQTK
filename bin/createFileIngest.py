#!/usr/bin/env python
"""
Create an ingest of files in a specified directory.
"""

import argparse
import glob
import json
import logging
import os.path as osp

from smqtk.representation import get_data_set_impls
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import bin_utils, plugin


def default_config():
    return {
        "data_set": plugin.make_config(get_data_set_impls)
    }


def cli_parser():
    description = "Add a set of local system files to a data set via " \
                  "explicit paths or shell-style glob strings."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Add debug messaged to output logging.')

    group_configuration = parser.add_argument_group("Configuration")
    group_configuration.add_argument('-c', '--config',
                                     help="Path to the JSON configuration file")
    group_configuration.add_argument('--output-config',
                                     help="Optional path to output a default "
                                          "JSON configuration file to. "
                                          "This output file should be modified "
                                          "and used for this executable.")

    parser.add_argument("input_files", metavar='GLOB', nargs='*')

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    bin_utils.initialize_logging(logging.getLogger(),
                                 logging.INFO - (10*args.verbose))
    log = logging.getLogger("main")

    # Merge loaded config with default
    config_loaded = False
    config = default_config()
    if args.config:
        if osp.isfile(args.config):
            with open(args.config, 'r') as f:
                config.update(json.load(f))
            config_loaded = True
        elif not osp.isfile(args.config):
            log.error("Configuration file path not valid.")
            exit(1)

    # output configuration dictionary when asked for.
    bin_utils.output_config(args.output_config, config, log, True)

    if not config_loaded:
        log.error("No configuration provided")
        exit(1)

    #: :type: smqtk.representation.DataSet
    ds = plugin.from_plugin_config(config['data_set'], get_data_set_impls)
    log.debug("Script arguments:\n%s" % args)

    def ingest_file(fp):
        ds.add_data(DataFileElement(fp))

    for f in args.input_files:
        f = osp.expanduser(f)
        if osp.isfile(f):
            ingest_file(f)
        else:
            log.debug("Expanding glob: %s" % f)
            for g in glob.glob(f):
                ingest_file(g)


if __name__ == '__main__':
    main()
