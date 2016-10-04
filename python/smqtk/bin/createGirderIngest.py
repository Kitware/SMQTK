"""
Add a number of Girder hosted files to a dataset.

Files are discovered by recursively looking under any folders and items
provided. Explicit file IDs may be additionally provided. We also take in an
optional new-line separated file list of folder, item and file IDs.

Tool Configuration Parameters
-----------------------------

    girder_api_root
        Root URL where Girder is being hosted.

    api_key
        Optional user API key to provide additional permissions if target data
        is not public.

    api_query_batch
        Number of elements to query for at a time when expanding folder and item
        contents. A batch size of 0 means all elements are queried for at once.

    dataset_insert_batch_size
        An optional number of elements to batch before adding to the configured
        data-set. If this is 0 or None, we will collect all elements before
        adding then to the configured data-set.

TODO: Add support for searching collections

"""

import collections
from itertools import repeat
import logging

import requests

from smqtk.representation import get_data_set_impls
from smqtk.representation.data_element.girder import GirderDataElement
from smqtk.utils import bin_utils, plugin
from smqtk.utils.girder import GirderTokenManager
from smqtk.utils.parallel import parallel_map
from smqtk.utils.url import url_join


################################################################################
# Component Functions

def cli_parser():
    parser = bin_utils.basic_cli_parser(__doc__)
    g_girder = parser.add_argument_group('Girder References')

    g_girder.add_argument('-F', '--folder',
                          nargs='*', default=[], metavar='FOLDER_ID',
                          help='Specify specific folder IDs')
    g_girder.add_argument('--folder-list', metavar='PATH',
                          help='Path to a new-line separated file of folder '
                               'IDs')

    g_girder.add_argument('-i', '--item',
                          nargs='*', default=[], metavar='ITEM_ID',
                          help='Specify specific item IDs')
    g_girder.add_argument('--item-list', metavar='PATH',
                          help='Path to a new-line separated file of item IDs')

    g_girder.add_argument('-f', '--file',
                          nargs='*', default=[], metavar='FILE_ID',
                          help='Specify specific file IDs')
    g_girder.add_argument('--file-list', metavar='PATH',
                          help='Path to a new-line separated file of file IDs')

    return parser


def default_config():
    return {
        'tool': {
            'girder_api_root': 'http://localhost:8080/api/v1',
            'api_key': None,
            'api_query_batch': 1000,
            'dataset_insert_batch_size': None,
        },
        'plugins': {
            'data_set': plugin.make_config(get_data_set_impls()),
        }
    }


def iterate_query(q, batch_size):
    """

    :param q: function taking an offset and limit parameter, returning an
        iterable of elements to yield.
    :type q: (int, int) -> __generator

    :param batch_size: Number of elements to get from a single query.
    :type batch_size: int | None

    :return: Elements yielded by nested function
    :rtype: __generator

    """
    query_again = True
    y_total = 0
    while query_again:
        y_this_query = 0
        for r in q(offset=y_total, limit=batch_size):
            yield r
            y_this_query += 1
        query_again = batch_size > 0 and (y_this_query == batch_size)
        y_total += y_this_query


def get_folder_subfolders(api_root, folder_id, tm, batch_size):
    """
    Iterate folder sub-folder IDs

    :type api_root: str
    :type folder_id: str
    :type tm: GirderTokenManager
    :type batch_size: int
    :rtype: __generator[str]
    """
    def q(offset, limit):
        r = requests.get(url_join(api_root, 'folder'),
                         params={'parentType': 'folder',
                                 'parentId': folder_id,
                                 'offset': offset,
                                 'limit': limit},
                         headers=tm.get_requests_header())
        r.raise_for_status()
        for f_model in r.json():
            yield f_model['_id']

    return iterate_query(q, batch_size)


def get_folder_items(api_root, folder_id, tm, batch_size):
    """
    Iterate folder item IDs

    :type api_root: str
    :type folder_id: str
    :type tm: GirderTokenManager
    :type batch_size: int
    :rtype: __generator[str]
    """
    def q(offset, limit):
        r = requests.get(url_join(api_root, 'item'),
                         params={'folderId': folder_id,
                                 'offset': offset,
                                 'limit': limit},
                         headers=tm.get_requests_header())
        r.raise_for_status()
        for i_model in r.json():
            yield i_model['_id']

    return iterate_query(q, batch_size)


def get_item_files(api_root, item_id, tm, batch_size):
    """
    Iterate item file IDs and content type

    :type api_root: str
    :type item_id: str
    :type tm: GirderTokenManager
    :type batch_size: int
    :rtype: __generator[str]
    """
    def q(offset, limit):
        r = requests.get(url_join(api_root, 'item', item_id, 'files'),
                         params={'offset': offset,
                                 'limit': limit},
                         headers=tm.get_requests_header())
        r.raise_for_status()
        for file_model in r.json():
            yield file_model['_id'], file_model['mimeType']

    return iterate_query(q, batch_size)


def find_girder_files(api_root, folder_ids, item_ids, file_ids,
                      api_key=None, query_batch=None):
    """
    Query girder for file IDs nested under folders and items, and yield
    GirderDataElement instances for each child file discovered.

    We first find nested folder and items under given folder IDs, then files
    under discovered items, finally yielding GirderDataElements for file IDs
    discovered.

    Data elements yielded are in order of file IDs given, to files under items
    given, to files under items discovered in folders in a breadth-first order.

    :param api_root: Root URL of the girder instance to call.
    :type api_root: str

    :param folder_ids: Iterable of Girder folder IDs to recursively collect file
        elements from.
    :type folder_ids: collection.Iterable[str]

    :param item_ids: Iterable of Girder item IDs to collect file elements from.
    :type item_ids: collection.Iterable[str]

    :param file_ids: Iterable of Girder file IDs to make elements of.
    :type file_ids: collection.Iterable[str]

    :param api_key: Optional user API key to use for authentication when
        accessing private data.
    :type api_key: None | str

    :param query_batch: Number of elements to query for at a time when expanding
        folder and item contents. A batch size of 0 means all elements are
        queried for at once.
    :type query_batch: int

    :return: Generator yielding GirderDataElement instances.
    :rtype: __generator[GirderDataElement]

    """
    log = logging.getLogger(__name__)
    tm = GirderTokenManager(api_root, api_key)
    # Get the token once before parallel requests
    tm.get_token()

    if query_batch is None:
        query_batch = 0

    # Could also do something with multi-threading/processing

    log.info("Yielding elements from file IDs")
    file_fifo = collections.deque(file_ids)
    while file_fifo:  # Just file IDs
        file_id = file_fifo.popleft()
        log.debug('-f %s', file_id)
        e = GirderDataElement(file_id, api_root, api_key)
        e.token_manager = tm  # because we already made one
        yield e

    log.info("Collecting files from items")
    item_fifo = collections.deque(item_ids)
    while item_fifo:
        item_id = item_fifo.popleft()
        log.debug('-i %s', item_id)
        for file_id, ct in get_item_files(api_root, item_id, tm, query_batch):
            log.debug('   -f %s', file_id)
            e = GirderDataElement(file_id, api_root, api_key)
            e._content_type = ct
            e.token_manager = tm
            yield e

    # Collect items from folders, then files from items.
    log.info("Collecting items from folders")
    folder_fifo = collections.deque(folder_ids)
    while folder_fifo:
        folder_id = folder_fifo.popleft()
        log.debug("-F %s", folder_id)

        # Find sub-folders
        folder_fifo.extend(get_folder_subfolders(api_root, folder_id, tm, query_batch))

        for item_id in get_folder_items(api_root, folder_id, tm, query_batch):
            log.debug('   -i %s', item_id)
            for file_id, ct in get_item_files(api_root, item_id, tm, query_batch):
                log.debug('      -f %s (%s)', file_id, ct)
                e = GirderDataElement(file_id, api_root, api_key)
                e._content_type = ct
                e.token_manager = tm
                yield e


################################################################################
# Main

def main():
    args = cli_parser().parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    api_root = config['tool']['girder_api_root']
    api_key = config['tool']['api_key']
    api_query_batch = config['tool']['api_query_batch']
    insert_batch_size = config['tool']['dataset_insert_batch_size']

    # Collect N folder/item/file references on CL and any files referenced.
    #: :type: list[str]
    ids_folder = args.folder
    #: :type: list[str]
    ids_item = args.item
    #: :type: list[str]
    ids_file = args.file

    if args.folder_list:
        with open(args.folder_list) as f:
            ids_folder.extend([fid.strip() for fid in f])
    if args.item_list:
        with open(args.item_list) as f:
            ids_item.extend([iid.strip() for iid in f])
    if args.file_list:
        with open(args.file_list) as f:
            ids_file.extend([fid.strip() for fid in f])

    #: :type: smqtk.representation.DataSet
    data_set = plugin.from_plugin_config(config['plugins']['data_set'],
                                         get_data_set_impls())

    batch = collections.deque()
    rps = [0]*7
    for e in find_girder_files(api_root, ids_folder, ids_item, ids_file,
                               api_key, api_query_batch):
        batch.append(e)
        if insert_batch_size and len(batch) >= insert_batch_size:
            data_set.add_data(*batch)
            batch.clear()
        bin_utils.report_progress(log.info, rps, 1.0)

    if batch:
        data_set.add_data(*batch)

    log.info('Done')


################################################################################
# Entry-point

if __name__ == '__main__':
    main()
