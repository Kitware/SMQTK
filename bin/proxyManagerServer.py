#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

Small script to start and host a ProxyManager server over a port.

This takes a simple configuration file that looks like the following:

[server]
port = <integer>
authkey = <string>

"""

import argparse
import json
import logging
import os.path as osp

from smqtk.utils import bin_utils
from smqtk.utils import ProxyManager


def default_config():
    return {
        "port": 5000,
        "authkey": "CHANGE_ME",
    }


def cli_parser():
    description = "Server for hosting proxy manager which hosts proxy " \
                   "object instances."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Add debugging log messages.')

    group_configuration = parser.add_argument_group("Configuration")
    group_configuration.add_argument('-c', '--config',
                                     default=None,
                                     help='Path to the JSON configuration '
                                          'file.')
    group_configuration.add_argument('--output-config',
                                     default=None,
                                     help='Optional path to output default '
                                          'JSON configuration to. '
                                          'This output file should be modified '
                                          'and used for this executable.')

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    llevel = logging.DEBUG if args.verbose else logging.INFO
    bin_utils.initialize_logging(logging.getLogger(), llevel)
    log = logging.getLogger("main")

    # Merge loaded config with default
    config = default_config()
    if args.config:
        if osp.isfile(args.config):
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        elif not osp.isfile(args.config):
            log.error("Configuration file path not valid.")
            exit(1)

    bin_utils.output_config(args.output_config, config, log, True)

    # Default config options for this util are valid for running, so no "has
    # config loaded check here.

    port = int(config['port'])
    authkey = str(config['authkey'])

    mgr = ProxyManager(('', port), authkey)
    mgr.get_server().serve_forever()


if __name__ == '__main__':
    main()
