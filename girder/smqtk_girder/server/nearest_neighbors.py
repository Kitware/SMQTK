from girder import logger
from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import Resource, filtermodel, getCurrentToken, getCurrentUser, getApiUrl, RestException
from girder.utility.model_importer import ModelImporter
from girder.constants import AccessType

from smqtk.representation.descriptor_set.postgres import PostgresDescriptorSet
from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.representation.data_element.girder import GirderDataElement
from smqtk.representation.key_value.memory import MemoryKeyValueStore
from smqtk.algorithms.nn_index.lsh import LSHNearestNeighborIndex

from .utils import localSmqtkFileIdFromName

import functools


setting = ModelImporter.model('setting')

class NearestNeighbors(Resource):
    def __init__(self):
        self.resourceName = 'smqtk_nearest_neighbors'
        self.route('GET', ('nn',), self.nearestNeighbors)


    @staticmethod
    def descriptorSetFromItem(item):
        """
        Get the descriptor set related to the item (its folder id).

        Note that this only works for top level items in the directory,
        meaning images must have been processed for the directory
        this item is in. Ideally, when processing images works recursively, this
        should recursively ascend the dir tree looking for the first .smqtk
        directory.

        :param item: Item to find the descriptor set for, usually the item that
            the user is performing the nearest neighbors search on.
        """
        # this assumes the parent directory of the item has been processed. i.e. subdirectories
        # won't work. this should be fixed and this should recursively ascend looking for .smqtk
        # TODO also no error checking whatsoever

        return PostgresDescriptorSet('descriptor_set_%s' % item['folderId'],
                                     db_name=setting.get('smqtk_girder.db_name'),
                                     db_host=setting.get('smqtk_girder.db_host'),
                                     db_user=setting.get('smqtk_girder.db_user'),
                                     db_pass=setting.get('smqtk_girder.db_pass'))


    @staticmethod
    def nearestNeighborIndex(item, user, descriptorSet):
        """
        Get the nearest neighbor index from a given item and descriptor set.

        :param item: Item to find the nn index from, usually the item that the
            user is performing the nearest neighbors search on.
        :param user: The owner of the .smqtk folder.
        :param descriptorSet: The relevant descriptor set.
        """
        folder = ModelImporter.model('folder')

        _GirderDataElement = functools.partial(GirderDataElement,
                                               api_root=getApiUrl(),
                                               token=getCurrentToken()['_id'])

        smqtkFolder = folder.createFolder(folder.load(item['folderId'], user=user), '.smqtk',
                                          reuseExisting=True)

        try:
            meanVecFileId = localSmqtkFileIdFromName(smqtkFolder, 'mean_vec.npy')
            rotationFileId = localSmqtkFileIdFromName(smqtkFolder, 'rotation.npy')
            hash2uuidsFileId = localSmqtkFileIdFromName(smqtkFolder, 'hash2uuids.pickle')
        except Exception:
            logger.warn('SMQTK files didn\'t exist for performing NN on %s' % item['_id'])
            return None

        # TODO Should these be Girder data elements? Unnecessary HTTP requests.
        functor = ItqFunctor(mean_vec_cache=_GirderDataElement(meanVecFileId),
                             rotation_cache=_GirderDataElement(rotationFileId))

        hash2uuidsKV = MemoryKeyValueStore(_GirderDataElement(hash2uuidsFileId))

        return LSHNearestNeighborIndex(functor, descriptorSet,
                                       hash2uuidsKV, read_only=True)


    @access.user
    @filtermodel('item', addFields=('smqtk_distance',))
    @autoDescribeRoute(Description('Find the nearest neighbors (items) of an item.')
                       .modelParam('itemId', model='item', level=AccessType.READ)
                       .param('limit', 'Number of neighbors to query for.', default=25))
    def nearestNeighbors(self, item, limit, params):
        limit = int(limit)
        desc_set = self.descriptorSetFromItem(item)
        nn_index = self.nearestNeighborIndex(item, getCurrentUser(), desc_set)

        if nn_index is None:
            raise RestException('Nearest neighbor index could not be found.')

        try:
            smqtk_uuid = item['meta']['smqtk_uuid']
            descriptor = desc_set.get_descriptor(smqtk_uuid)
        except KeyError:
            raise RestException('Unable to retrieve image descriptor for querying object.')

        neighbors, dists = nn_index.nn(descriptor, limit)
        uuid_dist = dict(zip([x.uuid() for x in neighbors], dists))

        smqtkFolder = ModelImporter.model('folder').load(item['folderId'], user=getCurrentUser())
        items = list(ModelImporter.model('folder').childItems(smqtkFolder, filters={'meta.smqtk_uuid': {
            '$in': [x.uuid() for x in neighbors]
        }}))

        for item in items:
            item['smqtk_distance'] = uuid_dist[item['meta']['smqtk_uuid']]

        return items
