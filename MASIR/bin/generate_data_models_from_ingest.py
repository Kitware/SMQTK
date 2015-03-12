#!/usr/bin/env python
# coding=utf-8
"""
Generate standard data mode files from the current ingest
"""

import logging

from masir import IngestManager
from masir.search.colordescriptor import \
    ColorDescriptor_CSIFT, \
    ColorDescriptor_TCH

import masir_config


logging.basicConfig(
    format="%(levelname)7s - %(asctime)s - %(name)s - %(message)s"
)
logging.getLogger().setLevel(logging.INFO)


def main():
    import optparse
    usage = "Usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    parser.add_option('-d', '--data-dir',
                      help="Override data directory to use.")
    parser.add_option('-w', '--work-dir',
                      help="Override work directory to use.")
    parser.add_option('-l', '--list-descriptors',
                      action='store_true', default=False,
                      help="List descriptors that can be generated.")
    parser.add_option('-e', '--exclude', action='append', default=[],
                      help="Specify descriptors to not use.")
    parser.add_option('-t', '--threads', type=int, default=None,
                      help="Number of threads to use. If not provided, we use "
                           "all available.")
    opts, args = parser.parse_args()

    log = logging.getLogger('main')
    data_dir = opts.data_dir or masir_config.DIR_DATA
    work_dir = opts.work_dir or masir_config.DIR_WORK

    # Descriptors to generate models for
    descriptors = {}
    d = ColorDescriptor_CSIFT(data_dir, work_dir)
    descriptors[d.name] = d
    d = ColorDescriptor_TCH(data_dir, work_dir)
    descriptors[d.name] = d

    log.info("Descriptors to exclude: %s", opts.exclude)

    if opts.list_descriptors:
        print "\nAvailable Descriptors:"
        for n in descriptors:
            print "\t%s" % n
        print
        exit(0)

    im = IngestManager(data_dir)
    log.info("%d items in current ingest", len(im))

    for name, descr in descriptors.iteritems():
        if name in opts.exclude:
            log.info("Skipping excluded descriptor %s", name)
            continue
        log.info("Generating data models for descriptor '%s'",
                 descr.__class__.__name__)
        descr.generate_feature_data(im, parallel=opts.threads)


if __name__ == '__main__':
    main()