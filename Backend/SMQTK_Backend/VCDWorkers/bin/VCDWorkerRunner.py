#!/usr/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Script to run a specified descriptor worker that inherits from
VCDWorkerInterface.

"""

import logging
import optparse
import os.path as osp
import sys

from SMQTK_Backend.VCDWorkers import VCDWorkerInterface


class MyParser (optparse.OptionParser):
    """ Overriding description formatter in OptionParser
    """
    def format_description(self, formatter):
        return self.description


# ======================================================================
def run_descriptor(opts, args):
    """
    Run a descriptor with the supplied options and parameters
    Options - a dictionary of, well, options
    worker_file => The path to the descriptor worker wrapper file.
    video_file => Path to the video file to process.
    verbosity => verbosity level
    name => class name [optional]

    The args
    """
    logging.basicConfig(format='%(levelname)7s - %(asctime)s - '
                               '%(name)s.%(funcName)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO - (10 * opts['verbosity']))
    LOG = logging.getLogger("DescriptorWorkerRunner")
    LOG.debug("Printing Debug messages")

    ###
    # Check required options
    #
    if not 'worker_file' in opts:
        raise ValueError("Require a path to a worker file to run.")
    elif not osp.exists(opts['worker_file']):
        raise ValueError("Provided worker file path did not point to an "
                         "existing file.")

    if not 'video_file' in opts:
        raise ValueError("Require a path to an input video file.")
    elif not osp.exists(opts['video_file']):
        raise ValueError("Provided input video file path did not point to an "
                         "existing file.")

    ###
    # Import the class type from the supplied file
    #

    # determine the name of the class to import
    classname = opts['name'] if 'name' in opts else \
        osp.splitext(osp.basename(opts['worker_file']))[0]
    LOG.debug("Extracting class by name: '%s'", classname)

    # add file containing directory to the path so that the import will find it
    sys.path.append(osp.dirname(opts['worker_file']))

    # the name to import will always be the basename of the file without the
    # extension
    module = __import__(osp.splitext(osp.basename(opts['worker_file']))[0])
    worker_class = getattr(module, classname)
    assert issubclass(worker_class, VCDWorkerInterface), \
        'Worker class is not a subclass of VCDWorkerInterface.'

    ###
    # Instantiate/run extracted class
    #
    LOG.info("Initializing worker '%s'", worker_class.__name__)
    LOG.debug('pointing to input video: %s', opts['video_file'])

    worker = worker_class(opts['video_file'], *args)
    LOG.info("Running worker '%s'", worker_class.__name__)

    worker.run()  # run the descriptor
    LOG.info("Worker '%s' complete", worker_class.__name__)


if __name__ == '__main__':
    description = """\
    General execution wrapper for descriptor worker implementations.

    Positional arguments may be provided to this script and they will be passed
    through to the constructor of the descriptor worker.
    """
    parser = MyParser(usage="%prog -f WORKER_FILE -i VIDEO_FILE "
                            "[-n NAME] [-v] [*pass_through_args]",
                      description=description)

    req_group = optparse.OptionGroup(parser, "Required Options")
    req_group.add_option('-f', '--file', dest='worker_file',
                         help="The path to the descriptor worker wrapper file.")
    req_group.add_option('-i', '--input-video', dest='video_file',
                         help='Path to the video file to process.')

    opt_group = optparse.OptionGroup(parser, "Optionals")
    opt_group.add_option('-n', '--name',
                         help='By default, we look for a class in the supplied '
                              'worker file that matches the name of the file '
                              '(minus the extension). If this option is '
                              'provided, we instead look for a class by this '
                              'name (case sensitive).')

    v_group = optparse.OptionGroup(parser, "Verbosity/Logging")
    v_group.add_option('-v', action='count', default=0, dest='verbosity',
                       help='Make output more verbose. By default, only '
                            'informational messages are shown. Additional '
                            'verbosity will show debugging messages')

    parser.add_option_group(req_group)
    parser.add_option_group(opt_group)
    parser.add_option_group(v_group)

    opts, args = parser.parse_args()

    # convert opts to dictionary format
    options = { }
    if opts.worker_file:  options['worker_file'] = opts.worker_file
    if opts.video_file:   options['video_file'] = opts.video_file
    if opts.verbosity:    options['verbosity'] = opts.verbosity
    if opts.name:         options['name'] = opts.name

    run_descriptor(options, args)
