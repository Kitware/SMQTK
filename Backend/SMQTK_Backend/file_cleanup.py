#!/bin/env python
"""

LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""


import optparse
import os
import os.path as osp
import pprint

from SMQTK_Backend.ECDController import ECDController
from SMQTK_Backend.SmqtkController import SmqtkController
from SMQTK_Backend.utils.SafeConfigCommentParser import SafeConfigCommentParser


def main():
    parser = optparse.OptionParser()
    parser.add_option('-c', '--config', action='append', default=[],
                      dest='config_files',
                      help="Configuration file(s) used for the SMQTK Backend "
                           "run we want to clean up from. If no configuration "
                           "files were given, we will clean-up based on "
                           "system default paths.")
    opts, args = parser.parse_args()

    config = SmqtkController.generate_config()

    for cfile in opts.config_files:
        c = SafeConfigCommentParser(cfile)
        config.update(c)

    if not opts.config_files:
        print "Using default options (no config files provided)"
    else:
        print "Using options from config files:"
        pprint.pprint(opts.config_files)

    # Currently, the only work directory used is in the ECDC
    ecdc_work_dir = config.get(ECDController.CONFIG_SECT, 'work_directory')
    print "Removing files from ECDController work directory:", ecdc_work_dir
    for f in os.listdir(ecdc_work_dir):
        print "-->", f
        os.remove(osp.join(ecdc_work_dir, f))


if __name__ == '__main__':
    main()
