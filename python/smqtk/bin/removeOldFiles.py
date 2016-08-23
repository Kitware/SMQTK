"""
Monitor files under a given directory, removing them from dist after an expiry
period.
"""

import argparse
import logging
import os
import time

from smqtk.utils.bin_utils import initialize_logging


sf_log = logging.getLogger("smqtk.scan_files")
is_log = logging.getLogger("smqtk.interval_scan")


def scan_files(base_dir, expire_seconds, action):
    """
    For files under the given starting directory, check the last access time
    against the expiry period (seconds) provided, applying action to that file
    path.

    :param base_dir: Starting directory for processing

    :param expire_seconds: Number of seconds since the last access of a file
        that should trigger the application of ``action``.

    :param action: Single argument function that will be given the path of a
        file that has not been accessed in ``expiry_seconds`` seconds.

    """
    for f in os.listdir(base_dir):
        f = os.path.join(base_dir, f)
        if os.path.isfile(f):
            s = os.stat(f)
            if time.time() - s.st_atime > expire_seconds:
                sf_log.debug("Action triggered for file: %s", f)
                action(f)
        elif os.path.isdir(f):
            scan_files(f, expire_seconds, action)
        else:
            raise RuntimeError("Encountered something not a file or directory? "
                               "Path: %s" % f)


def remove_file_action(filepath):
    os.remove(filepath)


def interval_scan(interval, base_dir, expire_seconds, action):
    """
    Action scan a directory every ``interval`` seconds. This will continue to
    run until the process is interrupted.

    :param interval: Number of seconds to wait in between each scan.

    :param base_dir: Starting directory for processing

    :param expire_seconds: Number of seconds since the last access of a file
        that should trigger the application of ``action``.

    :param action: Single argument function that will be given the path of a
        file that has not been accessed in ``expiry_seconds`` seconds.

    """
    while 1:
        is_log.debug("Starting scan on directory: %s", base_dir)
        scan_files(base_dir, expire_seconds, action)
        time.sleep(interval)


def cli_parser():
    description = "Utility to recursively scan and remove files underneath a " \
                  "given directory if they have not been modified for longer " \
                  "than a set amount of time."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--base-dir',
                        help='Starting directory for scan.')
    parser.add_argument('-i', '--interval', type=int,
                        help='Number of seconds between each scan (integer).')
    parser.add_argument('-e', '--expiry', type=int,
                        help='Number of seconds until a file has "expired" '
                             '(integer).')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Display more messages (debugging).')

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    logging_level = logging.INFO
    if args.verbose:
        logging_level = logging.DEBUG
    initialize_logging(logging.getLogger("smqtk"), logging_level)

    base_dir = args.base_dir
    interval_seconds = args.interval
    expiry_seconds = args.expiry

    interval_scan(interval_seconds, base_dir, expiry_seconds,
                  remove_file_action)


if __name__ == '__main__':
    main()
