#!/usr/bin/env python
"""
Utility for fetching remotely stored image data from the CDR ElasticSearch
instance.

Files will be transferred into the configured directory with the format::

    <output_dir>/<index>/<_type>/<id>.<type_extension>

Configuration Notes:

    image_types
        This is a list of image MIMETYPE suffixes to include when querying
        the ElasticSearch instance. If all types should be considered, this
        should be set to an empty list.

    stored_http_auth
        This is only used for stored-data URLs and only if both a username
        and password is given.

    elastic_search
        batch_size
            The number of query hits to fetch at a time from the instance.

ES Compatibility: 1.x, 2.x

"""

import datetime
import logging
import mimetypes
import os
import re

import certifi
import elasticsearch
import elasticsearch.helpers
import elasticsearch_dsl
from elasticsearch_dsl import Q
import requests

from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils.bin_utils import (
    basic_cli_parser,
    doc_as_description,
    report_progress,
    utility_main_helper,
)
from smqtk.utils.file_utils import safe_create_dir
from smqtk.utils.parallel import parallel_map


__author__ = "paul.tunison@kitware.com"


# Fix global MIMETYPE map
if '.jfif' in mimetypes.types_map:
    del mimetypes.types_map['.jfif']
if '.jpe' in mimetypes.types_map:
    del mimetypes.types_map['.jpe']


def cdr_images_after(es_instance, index, image_types, crawled_after=None,
                     inserted_after=None, agg_img_types=False,
                     domain='weapons'):
    """
    Return query and return an iterator over ES entries.

    Results yielded in ascending CDR insertion order (i.e. FIFO). This should
    cause slicing to be stable.

    :param es_instance: elasticsearch.Elasticsearch instance.
    :type es_instance:

    :param index: ElasticSearch index to draw from.
    :type index: str

    :param image_types: List of image content type suffixes
        (e.g. ['png', 'jpeg'])
    :type image_types:

    :param crawled_after: Optional timestamp to constrain query elements to
        only those collected/crawled after this time.
    :type crawled_after: datetime.datetime

    :param inserted_after: Optional timestamp to constrain query elements to
        only those inserted into the ES instance/index after this time.
    :param inserted_after: datetime.datetime

    :param agg_img_types: If we should add an aggregation on image types to the
        query (prevents scanning).

    :param domain: The _type to filter by. "weapons" by default.

    :return:
    :rtype:

    """
    log = logging.getLogger(__name__)
    log.info("Forming ES CDR image query for types: %s", image_types)

    base_search = elasticsearch_dsl.Search()\
        .using(es_instance)\
        .index(index)\
        .fields(['_id', '_timestamp', '_type',
                 'content_type', 'obj_original_url', 'obj_stored_url',
                 'timestamp', 'version',
                 ])

    if domain:
        base_search = base_search.doc_type(domain)

    # I think `_type` filter is redundant with `doc_type` specification above
    if elasticsearch_dsl.VERSION[0] == 2:
        # ES 2.x version
        f = Q('term', version='2.0') \
            & Q('term', content_type='image')
        if image_types:
            f &= Q('terms', content_type=image_types)
        if domain:
            log.debug("Constraining _type: %s", domain)
            f &= Q('term', _type=domain)
        if crawled_after:
            log.debug("Constraining to entries crawled after: %s",
                      crawled_after)
            f &= Q('range', timestamp={'gt': crawled_after})
        if inserted_after:
            log.debug("Constraining to entries inserted after: %s",
                      inserted_after)
            f &= Q('range', _timestamp={'gt': inserted_after})
    else:
        # ES 1.x version
        from elasticsearch_dsl.filter import F
        f = F('term', version='2.0') \
            & F('term', content_type='image')
        if image_types:
            f &= F('terms', content_type=image_types)
        if domain:
            log.debug("Constraining _type: %s", domain)
            f &= F('term', _type=domain)
        if crawled_after:
            log.debug("Constraining to entries crawled after: %s",
                      crawled_after)
            f &= F('range', timestamp={'gt': crawled_after})
        if inserted_after:
            log.debug("Constraining to entries inserted after: %s",
                      inserted_after)
            f &= F('range', _timestamp={'gt': inserted_after})

    q = base_search\
        .filter(f)\
        .sort({'_timestamp': {"order": "asc"}})

    if agg_img_types:
        log.debug("Aggregating image content type information")
        q.aggs.bucket('per_type', 'terms', field='content_type')

    return q


def try_download(uri, auth=None):
    """
    Attempt URI download via get, return success and request instance if
    successful, or the optional error that occurred if false.
    """
    if uri:
        try:
            r = requests.get(uri, auth=auth)
            if r.ok:
                return True, r
        except Exception, ex:
            return False, ex
    return False, None


def fetch_cdr_query_images(q, output_dir, scan_record, cores=None,
                           stored_http_auth=None, batch_size=1000):
    """
    Queries for and saves image content underneath a nested directory from the
    given output directory.

    save location: <output_dir>/<index>/<type>/<id>.<extension>

    :param q: Query return from :func:`cdr_images_after`
    :type q:

    :param output_dir: Root output directory.
    :type output_dir:

    :param scan_record: Path to the file to write scan records to. We write out
        in CSV format (',' delimiter).

    :param cores: number of multiprocessing cores to use for asynchronous data
        downloading.

    :param stored_http_auth: Optional HTTP authentication username, password
        pair tuple to use when fetching the stored image data. This will not be
        applied when fetching from the original data URL.
    :type stored_http_auth: None | (str, str)

    :param batch_size: Number of hits returned by a single execute during
        result iteration.
    :type batch_size: int

    """
    log = logging.getLogger(__name__)
    log.info("Starting CDR image fetch")

    # DL record info :: [ CDR ID, local image path, SMQTK UUID ]
    m = mimetypes.MimeTypes()

    def dl_image(meta):
        try:
            c_type = meta['fields']['content_type'][0]
            obj_stored_url = meta['fields']['obj_stored_url'][0]
            obj_original_url = meta['fields']['obj_original_url'][0]

            c_ext = m.guess_extension(c_type, strict=False)
            if c_ext is None:
                log.warn("Guessed 'None' extension for content-type '%s', "
                         "skipping.", c_type)
                return None

            save_dir = os.path.abspath(os.path.expanduser(
                os.path.join(output_dir, meta['index'], meta['doc_type'])
            ))
            save_file = meta['id'] + c_ext
            save_path = os.path.join(save_dir, save_file)

            # Save/write file if needed
            if not os.path.isfile(save_path):
                # First try 'stored' url, fallback on original
                # Return None if failed to download anything
                ok, r = try_download(obj_stored_url, stored_http_auth)
                if not ok:
                    log.warn("Failed to download stored-data URL \"%s\" "
                             "(error=%s)",
                             obj_stored_url, str(r))

                    ok, r = try_download(obj_original_url)
                    if not ok:
                        log.warn("Failed to download original URL \"%s\" "
                                 "(error=%s)",
                                 obj_stored_url, str(r))
                        return None

                # Assuming OK at this point
                content = r.content

                d = DataMemoryElement(content, c_type)

                safe_create_dir(save_dir)
                with open(save_path, 'wb') as out:
                    log.debug("Saving to file: '%s'", save_path)
                    out.write(content)
            else:
                d = DataFileElement(save_path)

            return meta['id'], save_path, d.uuid()
        except KeyError, ex:
            log.error("Failed to find key %s in meta block: %s",
                      str(ex), meta)
            raise

    # def iter_scan_meta():
    #     q_scan = q.params(size=batch_size)
    #     restart = True
    #     i = 0
    #     while restart:
    #         restart = False
    #         try:
    #             log.debug("Starting scan from index %d", i)
    #             for h in q_scan.scan():
    #                 # noinspection PyProtectedMember
    #                 yield h.meta._d_
    #                 # Index of the next element to yield if scan fails in
    #                 # next iteration.
    #                 i += 1
    #         except elasticsearch.ConnectionTimeout, ex:
    #             log.warning("ElasticSearch timed out (error = %s)", str(ex))
    #             restart = True
    #             log.debug("Restarting query from index %d", i)
    #             q_scan = q_scan[i:]
    #         except elasticsearch.helpers.ScanError, ex:
    #             log.warning("ElasticSearch scan scan exception (error = %s)",
    #                         str(ex))
    #             restart = True
    #             log.debug("Restarting query from index %d", i)
    #             q_scan = q_scan[i:]

    def iter_scan_meta():
        """
        Using batch iteration vs. scan function. Tested speed difference was
        negligible and this allows for more confident ordering on restart
        """
        restart = True
        i = 0
        total = q[0:0].execute().hits.total
        while restart:
            restart = False
            try:
                log.debug("Starting ordered scan from index %d", i)
                while i < total:
                    b_start = i
                    b_end = i + batch_size
                    for h in q[b_start:b_end].execute():
                        # noinspection PyProtectedMember
                        yield h.meta._d_
                        i += 1
            except elasticsearch.ConnectionTimeout, ex:
                log.warning("ElasticSearch timed out (error = %s)", str(ex))
                restart = True
                log.debug("Restarting query from index %d", i)

    log.info("Initializing image download/record parallel iterator")
    img_dl_records = parallel_map(
        dl_image, iter_scan_meta(),
        name='image_download',
        use_multiprocessing=True,
        cores=cores
    )

    # Write out
    log.info("Starting iteration/file-write")
    rp_state = [0] * 7
    with open(scan_record, 'w') as record_file:
        for r in img_dl_records:
            if r is not None:
                cdr_id, local_path, uuid = r
                record_file.write('%s,%s,%s\n'
                                  % (cdr_id, local_path, uuid))
            report_progress(log.debug, rp_state, 1.0)
        # Final report
        rp_state[1] -= 1
        report_progress(log.debug, rp_state, 0)


def default_config():
    return {
        "image_types": ['jpeg', 'png', 'tiff'],
        "elastic_search": {
            "instance_address": "CHANGEME",
            "index": "CHANGEME",
            "username": "CHANGEME",
            "password": "CHANGEME",
            "batch_size": 10000,
        },
        "stored_http_auth": {
            'name': None,
            'pass': None,
        },
        "parallel": {
            "cores": None,
        }
    }


def cli_parser():
    """
    :rtype: argparse.ArgumentParser
    """
    parser = basic_cli_parser(doc_as_description(__doc__))

    parser.add_argument('-s', '--report-size',
                        action='store_true', default=False,
                        help="Report the number of elements that would be "
                             "scanned by the ElasticSearch query generated "
                             "and then exit.")
    parser.add_argument('--crawled-after',
                        default=None,
                        help="Optional timestamp constraint to only get "
                             "content that was crawled after the given time. "
                             "Time should be in UTC."
                             "Timestamp format like: '2016-01-01T12:00:00Z'")
    parser.add_argument('--inserted-after',
                        default=None,
                        help="Optional timestamp constraint to only get "
                             "content that was inserted into the "
                             "ElasticSearch instance/index after the given "
                             "time. Time should be in UTC."
                             "Timestamp format like: '2016-01-01T12:00:00Z'")

    g_output = parser.add_argument_group("Output")
    g_output.add_argument('-d', '--output-dir',
                          metavar='PATH',
                          help='Output image directory path.')
    g_output.add_argument('-l', '--file-list',
                          metavar='PATH',
                          help='Path to an output CSV file where downloaded '
                               'files are recorded along with their '
                               'associated CDR identifier as SHA1 checksum.')

    return parser


def main():
    args = cli_parser().parse_args()
    config = utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    report_size = args.report_size
    crawled_after = args.crawled_after
    inserted_after = args.inserted_after

    #
    # Check config properties
    #
    m = mimetypes.MimeTypes()
    # non-strict types (see use of ``guess_extension`` above)
    m_img_types = set(m.types_map_inv[0].keys() + m.types_map_inv[1].keys())
    if not isinstance(config['image_types'], list):
        raise ValueError("The 'image_types' property was not set to a list.")
    for t in config['image_types']:
        if ('image/' + t) not in m_img_types:
            raise ValueError("Image type '%s' is not a valid image MIMETYPE "
                             "sub-type." % t)

    if not report_size and args.output_dir is None:
        raise ValueError("Require an output directory!")
    if not report_size and args.file_list is None:
        raise ValueError("Require an output CSV file path!")

    #
    # Initialize ElasticSearch stuff
    #
    es_auth = None
    if config['elastic_search']['username'] and config['elastic_search']['password']:
        es_auth = (config['elastic_search']['username'],
                   config['elastic_search']['password'])

    es = elasticsearch.Elasticsearch(
        config['elastic_search']['instance_address'],
        http_auth=es_auth,
        use_ssl=True, verify_certs=True,
        ca_certs=certifi.where(),
    )

    #
    # Query and Run
    #
    http_auth = None
    if config['stored_http_auth']['name'] and config['stored_http_auth']['pass']:
        http_auth = (config['stored_http_auth']['name'],
                     config['stored_http_auth']['pass'])

    ts_re = re.compile('(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z')
    if crawled_after:
        m = ts_re.match(crawled_after)
        if m is None:
            raise ValueError("Given 'crawled-after' timestamp not in correct "
                             "format: '%s'" % crawled_after)
        crawled_after = datetime.datetime(*[int(e) for e in m.groups()])
    if inserted_after:
        m = ts_re.match(inserted_after)
        if m is None:
            raise ValueError("Given 'inserted-after' timestamp not in correct "
                             "format: '%s'" % inserted_after)
        inserted_after = datetime.datetime(*[int(e) for e in m.groups()])

    q = cdr_images_after(es, config['elastic_search']['index'],
                         config['image_types'], crawled_after, inserted_after)

    log.info("Query Size: %d", q[0:0].execute().hits.total)
    if report_size:
        exit(0)

    fetch_cdr_query_images(q, args.output_dir, args.file_list,
                           cores=int(config['parallel']['cores']),
                           stored_http_auth=http_auth,
                           batch_size=int(config['elastic_search']['batch_size']))


if __name__ == '__main__':
    main()
