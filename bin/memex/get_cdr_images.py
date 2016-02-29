"""

ES Compatibility: 1.x

"""

import logging
import mimetypes
import os

import certifi
import elasticsearch
import elasticsearch_dsl
from elasticsearch_dsl import Q
import requests

from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils.bin_utils import report_progress
from smqtk.utils.file_utils import safe_create_dir
from smqtk.utils.parallel import parallel_map


__author__ = "paul.tunison@kitware.com"


# Fix global MIMETYPE map
if '.jfif' in mimetypes.types_map:
    del mimetypes.types_map['.jfif']
if '.jpe' in mimetypes.types_map:
    del mimetypes.types_map['.jpe']


# def now_utc_datetime():
#     import datetime
#     import time
#
#     t = time.time()
#     s = time.gmtime(t)
#     now = datetime.datetime(s.tm_year, s.tm_mon, s.tm_mday,
#                             s.tm_hour, s.tm_min, s.tm_sec,
#                             int((t - int(t)) * 1000))
#     return now


ES_INSTANCE = 'CHANGE ME'
ES_USER = 'CHANGE ME'
ES_PASS = 'CHANGE ME'


es = elasticsearch.Elasticsearch(
    ES_INSTANCE,
    http_auth=(ES_USER, ES_PASS),
    use_ssl=True, verify_certs=True,
    ca_certs=certifi.where(),
)


def cdr_images_after(es_instance, image_types, after_date=None,
                     agg_img_types=False, domain='weapons'):
    """
    Return query and return an iterator over ES entries.

    Results yielded in ascending CDR insertion order (i.e. FIFO). This should
    cause slicing to be stable.

    :param es_instance: elasticsearch.Elasticsearch instance.
    :type es_instance:

    :param image_types: List of image content type suffixes
        (e.g. ['png', 'jpeg'])
    :type image_types:

    :param after_date:
    :type after_date:

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
        .index('memex-domains')\
        .fields(['_id', '_timestamp', '_type',
                 'content_type', 'obj_original_url', 'obj_stored_url',
                 'timestamp', 'version',
                 ])\

    if domain:
        base_search = base_search.doc_type(domain)

    # I think `_type` filter is redundant with `doc_type` specification above
    if elasticsearch_dsl.VERSION[0] == 2:
        # ES 2.x version
        f = Q('term', version='2.0') \
            & Q('term', content_type='image') \
            & Q('terms', content_type=image_types)
        if domain:
            log.debug("Constraining _type: %s", domain)
            f &= Q('term', _type=domain)
        if after_date:
            log.debug("Constraining _timestamp to after: %s", after_date)
            f &= Q('range', _timestamp={'gt': after_date})
    else:
        # ES 1.x version
        from elasticsearch_dsl.filter import F
        f = F('term', version='2.0') \
            & F('term', content_type='image') \
            & F('terms', content_type=image_types)
        if domain:
            log.debug("Constraining _type: %s", domain)
            f &= F('term', _type=domain)
        if after_date:
            log.debug("Constraining _timestamp to after: %s", after_date)
            f &= F('range', _timestamp={'gt': after_date})

    q = base_search\
        .filter(f)\
        .sort({'_timestamp': {"order": "asc"}})

    if agg_img_types:
        log.debug("Aggregating image content type information")
        q.aggs.bucket('per_type', 'terms', field='content_type')

    return q


def fetch_cdr_query_images(q, output_dir, scan_record, cores=None):
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

    """
    log = logging.getLogger(__name__)
    log.info("Starting CDR image fetch")

    # DL record info :: [ CDR ID, local image path, SMQTK UUID ]
    m = mimetypes.MimeTypes()

    def try_download(uri):
        try:
            r = requests.get(uri)
            if r.ok:
                return True, r
        except Exception, ex:
            return False, ex
        return False, None

    def dl_image(meta):
        c_type = meta['fields']['content_type'][0]
        obj_stored_url = meta['fields']['obj_stored_url'][0]
        obj_original_url = meta['fields']['obj_original_url'][0]

        c_ext = m.guess_extension(c_type)
        save_dir = os.path.abspath(os.path.expanduser(
            os.path.join(output_dir, meta['index'], meta['doc_type'])
        ))
        save_file = meta['id'] + c_ext
        save_path = os.path.join(save_dir, save_file)

        # Save/write file if needed
        if not os.path.isfile(save_path):
            # First try 'stored' url, fallback on original
            # Return None if failed to download anything
            ok, r = try_download(obj_stored_url)
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
                out.write(content)
        else:
            d = DataFileElement(save_path)

        return meta['id'], save_path, d.uuid()

    def iter_scan_meta():
        # for h in q.scan():
        #     # noinspection PyProtectedMember
        #     yield h.meta._d_

        q_scan = q
        timed_out = False
        i = 0
        while timed_out:
            timed_out = False
            try:
                for h in q_scan.scan():
                    # noinspection PyProtectedMember
                    yield h.meta._d_
                    # Index of the next element to yield if scan fails in next
                    # iteration.
                    i += 1
            except elasticsearch.ConnectionTimeout:
                log.warning("ElasticSearch timed out, restarting from index "
                            "%d", i)
                timed_out = True
                q_scan = q_scan[i:]

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


def main():
    description = """
    Utility for fetching remotely stored image data from the CDR ElasticSearch
    instance.

    Files will be transferred into the configured directory with the format::

        <output_dir>/<index>/<_type>/<id>.<type_extension>

    """


from smqtk.utils.bin_utils import logging, initialize_logging
if not logging.getLogger('smqtk').handlers:
    initialize_logging(logging.getLogger('smqtk'), logging.DEBUG)
    initialize_logging(logging.getLogger('__main__'), logging.DEBUG)

q = cdr_images_after(es, ['jpeg', 'png'])
# fetch_cdr_query_images(q, 'test_output', 'test_output.csv')
