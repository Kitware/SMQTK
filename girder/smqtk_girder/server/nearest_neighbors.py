from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import Resource, filtermodel, getCurrentToken
from girder.utility.model_importer import ModelImporter

from .settings import DB_NAME, DB_HOST, DB_USER, DB_PASS, GIRDER_API_ROOT, JOB_API_KEY

from girder_client import GirderClient, HttpError

from girder.constants import AccessType, TokenScope

from smqtk.representation.descriptor_index.postgres import PostgresDescriptorIndex
from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.representation.data_element.girder import GirderDataElement
from smqtk.representation.key_value.memory import MemoryKeyValueStore
from smqtk.algorithms.nn_index.lsh import LSHNearestNeighborIndex

from .utils import getCreateFolder, smqtkFileIdFromName

class NearestNeighbors(Resource):
    def __init__(self):
        self.resourceName = 'smqtk_nearest_neighbors'
        self.route('GET', ('nn',), self.nearestNeighbors)


    @staticmethod
    def descriptorIndexFromItem(item):
        """
        Get the descriptor index related to the item (its folder id).

        Note that this only works for top level items in the directory,
        meaning images must have been processed for the directory
        this item is in. Ideally, when processing images works recursively, this
        should recursively ascend the dir tree looking for the first .smqtk
        directory.

        :param item: Item to find the descriptor index for, usually the item that
            the user is performing the nearest neighbors search on.
        """
        # this assumes the parent directory of the item has been processed. i.e. subdirectories
        # won't work. this should be fixed and this should recursively ascend looking for .smqtk
        # TODO also no error checking whatsoever
        return PostgresDescriptorIndex('descriptor_index_%s' % item['folderId'],
                                       db_name=DB_NAME,
                                       db_host=DB_HOST,
                                       db_user=DB_USER,
                                       db_pass=DB_PASS)


    @staticmethod
    def nearestNeighborIndex(item, descriptorIndex):
        """
        Get the nearest neighbor index from a given item and descriptor index.

        :param item: Item to find the nn index from, usually the item that the
            user is performing the nearest neighbors search on.
        :param descriptorIndex: The relevant descriptor index.
        """
        folder = ModelImporter.model('folder')

        _GirderDataElement = functools.partial(GirderDataElement,
                                               api_root=GIRDER_API_ROOT,
                                               token=getCurrentToken()['_id'])

        smqtkFolder = folder.createFolder(folder.load(item['parentId']), '.smqtk',
                                          reuseExisting=True)

        try:
            meanVecFileId = next(folder.childItems(smqtkFolder,
                                                   filters={'name': 'mean_vec.npy'}))['_id']
            rotationFileId = next(folder.childItems(smqtkFolder,
                                                    filters={'name': 'rotation.npy'}))['_id']
            hash2uuidsFileId = next(folder.childItems(smqtkFolder,
                                                      filters={'name': 'hash2uuids.pickle'}))['_id']
        except Exception:
            # TODO Log message about these files not existing
            return None

        # TODO Should these be Girder data elements? Unnecessary HTTP requests.
        functor = ItqFunctor(mean_vec_cache=_GirderDataElement(meanVecFileId),
                             rotation_cache=_GirderDataElement(rotationFileId))

        hash2uuidsKV = MemoryKeyValueStore(_GirderDataElement(hash2uuidsFileId))

        return LSHNearestNeighborIndex(functor, descriptorIndex,
                                       hash2uuidsKV, read_only=True, live_reload=True)


    @access.user
    @filtermodel('item')
    @autoDescribeRoute(Description('Find the nearest neighbors (items) of an item.')
                       .modelParam('itemId', model='item', level=AccessType.READ)
                       .param('limit', 'Number of neighbors to query for.', default=25))
    def nearestNeighbors(self, item, limit, params):
        limit = int(limit)
        desc_index = self.descriptorIndexFromItem(item)
        nn_index = self.nearestNeighborIndex(item, desc_index)

        if nn_index is None:
            pass # TODO Raise HTTP error

        try:
            smqtk_uuid = item['meta']['smqtk_uuid']
            descriptor = desc_index.get_descriptor(smqtk_uuid)
        except KeyError:
            return {} # raise http error

        neighbors, dists = nn_index.nn(descriptor, limit)
        uuid_dist = dict(zip([x.uuid() for x in neighbors], dists))

        items = ModelImporter.model('item').find({
            'meta.smqtk_uuid': {
                '$in': [x.uuid() for x in neighbors]
            }})

        for item in items:
            item['smqtk_distance'] = uuid_dist[item['meta']['smqtk_uuid']]

        return items
