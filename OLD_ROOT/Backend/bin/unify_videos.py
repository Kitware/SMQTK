#!/usr/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Script to recursively look in one or more directories for video files and add
them to a unified structure whose base location is defined by the user.

This assumes the HVC format of files, i.e. files that look like 'HVC012345.mp4'.
The file must start with a 3 letter prefix and have an ID that consist of at
least 3 numbers. The unified directory structure will look like

Usage:
    ./unify_videos.py [options] directory1 [directory2 [...]]

"""
__author__ = 'paul.tunison'

import logging
import os
import os.path as osp
import re


def main():
    import optparse
    usage = "./%prog [options] DIR [DIR [...]]"
    description = ("Script to recursively look in one or more directories for "
                   "video files and add them to a unified structure whose base "
                   "location is defined by the user.")
    parser = optparse.OptionParser(usage=usage, description=description)
    parser.add_option('-v', '--verbosity',
                      action="count", default=0,
                      help="Make output more verbose. One '-v' adds "
                           "informational output and 2 adds debug output")
    parser.add_option('-o', '--output-dir',
                      default=osp.join(os.getcwd(), "ALL_VIDEOS"),
                      help="The directory in which to construct the unified "
                           "structure and place discovered video files. By "
                           "default this will find or create an 'ALL_VIDEOS' "
                           "directory in your current directory.")
    parser.add_option('-e', '--extension',
                      action='append', default=['mp4'], dest='extensions',
                      help="Also look for files with this extension. Leave out "
                           "the '.' when specifying extensions. The 'mp4' "
                           "extension will always be considered.")
    opts, args = parser.parse_args()

    ### Setting up state vars
    fmter = logging.Formatter(fmt='%(levelname)8s - %(asctime)s - '
                                  '%(name)s.%(funcName)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmter)

    root_log = logging.getLogger()
    root_log.addHandler(stream_handler)
    root_log.setLevel(logging.WARN - (10 * opts.verbosity))

    main_log = logging.getLogger('MAIN')

    search_dirs = tuple(osp.abspath(p) for p in args)
    output_dir = osp.abspath(opts.output_dir)
    valid_extensions = tuple('.%s' % e for e in opts.extensions)

    if not osp.isdir(output_dir):
        main_log.info("Creating '%s' unification directory", output_dir)
        os.makedirs(output_dir)

    ### Iterate through files in each directory,
    def iter_files(d):
        for p in (osp.join(d, e) for e in sorted(os.listdir(d))):
            if osp.isdir(p):
                for p2 in iter_files(p):
                    yield p2
            elif osp.splitext(p)[1] in valid_extensions:
                yield p

    pfx_regex = re.compile("\w{3}(\d{3})\d*\..+$")
    key_regex = re.compile("\w{3}(\d{3,})\..+$")

    main_log.info("Searching directories and linking files...")
    for d in search_dirs:
        main_log.debug("Entering directory: %s", d)
        for f in iter_files(osp.abspath(d)):
            main_log.debug(">>Examining file: %s", f)
            basename = osp.basename(f)
            pfx = pfx_regex.match(basename).group(1)
            main_log.debug("..... Prefix: %s", pfx)
            key = key_regex.match(basename).group(1)
            main_log.debug("..... Key   : %s", key)
            target = osp.join(output_dir, pfx, key, basename)
            main_log.debug("..... Target link: %s", target)
            if not osp.isdir(osp.dirname(target)):
                main_log.debug("..... Creating link directory")
                os.makedirs(osp.dirname(target))
            os.symlink(f, target)

    main_log.info("Done")


if __name__ == "__main__":
    main()