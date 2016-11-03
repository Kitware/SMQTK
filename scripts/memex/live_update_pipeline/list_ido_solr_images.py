#!/usr/bin/env python

import logging
import os

import solr

from smqtk.utils import (
    bin_utils,
    file_utils,
)


def solr_image_paths(solr_addr, begin_time, end_time, username, password, batch_size):
    log = logging.getLogger(__name__)

    conn = solr.Solr(solr_addr, http_user=username, http_pass=password)
    # Query for number of matching documents
    q = 'mainType:image AND indexedAt:[%s TO %s]' % (begin_time, end_time)
    r = conn.select(q, fields=['id'], rows=0)

    num_results = r.numFound
    log.debug("Found: %d", num_results)
    loops = (num_results // batch_size) + (num_results % batch_size > 0)
    log.debug("Making %d iterations", loops)

    for i in xrange(loops):
        r = conn.select(q, fields=['id'], rows=batch_size,
                        start=i * batch_size)
        for doc in r.results:
            yield doc['id'][5:]


def default_config():
    return {
        'solr_address': 'http://imagecat.dyndns.org/solr/imagecatdev',
        'solr_username': None,
        'solr_password': None,
        'batch_size': 100,
    }


def cli_parser():
    """
    :type parser: argparse.ArgumentParser
    :rtype: argparse.ArgumentParser
    """
    parser = bin_utils.basic_cli_parser(bin_utils.doc_as_description(__doc__))

    parser.add_argument('--after-time', metavar='TIMESTAMP',
                        help='Optional timestamp to constrain that we look '
                             'for entries added after the given time stamp. '
                             'This should be formatted according to Solr\'s '
                             'timestamp format (e.g. '
                             '"2016-01-01T00:00:00.000Z"). Sub-seconds are '
                             'optional. Timestamps should be in UTC (Zulu). '
                             'The constraint is inclusive.')
    parser.add_argument('--before-time', metavar='TIMESTAMP',
                        help='Optional timestamp to constrin that we look for '
                             'entries added before the given time stamp. See '
                             'the description of `--after-time` for format '
                             'info.')

    g_required = parser.add_argument_group("Required options")
    g_required.add_argument('-p', '--paths-file',
                            help='Path to the file to output collected file '
                                 'paths to (local filesystem path).')
    return parser


def main():
    description = """
    Utility for fetching remotely stored image paths from the JPL Solr index.

    Files will be transferred with their entire containing directories. For
    example, if the file was stored in "/data/things/image.png" remotely, it
    will be transferred locally to "<output_dir>/data/things/image.png".

    Assumptions:
        - JPL MEMEX Solr index key structure
            - `id` == "file:<abs-filepath>"
            - `mainType` is the first component of the MIMETYPE
            - `indexedAt` timestamp
    """
    args = cli_parser().parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    paths_file = args.paths_file
    after_time = args.after_time
    before_time = args.before_time

    #
    # Check dir/file locations
    #
    if paths_file is None:
        raise ValueError("Need a file path to to output transferred file "
                         "paths!")

    file_utils.safe_create_dir(os.path.dirname(paths_file))

    #
    # Start collection
    #
    remote_paths = solr_image_paths(
        config['solr_address'],
        after_time or '*', before_time or '*',
        config['solr_username'], config['solr_password'],
        config['batch_size']
    )

    log.info("Writing file paths")
    s = [0] * 7
    with open(paths_file, 'w') as of:
        for rp in remote_paths:
            of.write(rp + '\n')
            bin_utils.report_progress(log.info, s, 1.)
    # Final report
    s[1] -= 1
    bin_utils.report_progress(log.info, s, 0)


if __name__ == '__main__':
    main()
