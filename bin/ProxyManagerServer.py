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

import logging
import os.path as osp

from smqtk.utils import bin_utils
from smqtk.utils import SafeConfigCommentParser
from smqtk.utils.ProxyManager import ProxyManager


def main():
    parser = bin_utils.SMQTKOptParser()
    parser.add_option('-c', '--config', type=str,
                      help='Path to the configuration file.')
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                      help='Add debugging log messages.')
    opts, args = parser.parse_args()

    bin_utils.initializeLogging(logging.getLogger(),
                                logging.INFO - (10*opts.verbose))

    config_file = opts.config
    assert config_file is not None, \
        "Not given a configuration file for the server!"
    assert osp.exists(config_file), \
        "Given config file path does not exist."
    assert not osp.isdir(config_file), \
        "Given config file is a directory!"

    config = SafeConfigCommentParser()
    parsed = config.read(config_file)
    assert parsed, "Configuration file not parsed!"
    section = 'server'
    assert config.has_section(section), \
        "No server section found!"
    assert config.has_option(section, 'port'), \
        "No port option in config!"
    assert config.has_option(section, 'authkey'), \
        "No authkey option in config!"
    port = config.getint(section, 'port')
    authkey = config.get(section, 'authkey')

    mgr = ProxyManager(('', port), authkey)
    mgr.get_server().serve_forever()


if __name__ == '__main__':
    main()
