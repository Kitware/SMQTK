import json
import traceback

from bson.objectid import ObjectId

from girder import logger
from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import Resource, getCurrentUser, filtermodel, loadmodel, RestException
from girder.constants import AccessType
from girder.utility.model_importer import ModelImporter

from smqtk.algorithms.nn_index.lsh import LSHNearestNeighborIndex
from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.iqr import IqrSession, IqrController
from smqtk.representation.descriptor_index.postgres import PostgresDescriptorIndex
from smqtk.representation.key_value.memory import MemoryKeyValueStore

from .utils import getCreateSessionsFolder, localSmqtkFileIdFromName, smqtkDataElementFromGirderFileId


class Iqr(Resource):
    positive_seed_neighbors = 500

    def __init__(self):
        self.resourceName = 'smqtk_iqr'
        self.route('GET', ('session', ':id'), self.getSession)
        self.route('POST', ('session',), self.createSession)
        self.route('POST', ('refine', ':id'), self.refine)
        self.route('GET', ('results',), self.results)

        # Record of trained classifiers for a session. Session classifier
        # modifications locked under the parent session's global lock.
        #: :type: dict[collections.Hashable, SupervisedClassifier | None]
        self.session_classifiers = {}
        # Control for knowing when a new classifier should be trained for a
        # session (True == train new classifier). Modification for specific
        # sessions under parent session's lock.
        #: :type: dict[collections.Hashable, bool]
        self.session_classifier_dirty = {}
        self.controller = IqrController(False)

    @staticmethod
    def _descriptorIndexFromSessionId(sid):
        """
        Return the PostgresDescriptorIndex object from a given session id.

        This essentially does the postfixing of the data folder's ID to
        form the table name.

        :param sid: ID of the session
        :returns: Descriptor index representing the data folder related
        to the sid or None if no session exists
        :rtype: PostgresDescriptorIndex|None
        """
        session = ModelImporter.model('item').findOne({'_id': ObjectId(sid)})

        if not session:
            return None
        else:
            setting = ModelImporter.model('setting')
            return PostgresDescriptorIndex('descriptor_index_%s' % session['meta']['data_folder_id'],
                                           db_name=setting.get('smqtk_girder.db_name'),
                                           db_host=setting.get('smqtk_girder.db_host'),
                                           db_user=setting.get('smqtk_girder.db_user'),
                                           db_pass=setting.get('smqtk_girder.db_pass'))


    @staticmethod
    def _nearestNeighborIndex(sid, descriptor_index):
        """
        Retrieve the Nearest neighbor index for a given session.

        :param sid: ID of the session
        :param descriptor_index: The descriptor index corresponding to the session id,
        see _descriptorIndexFromSessionId.
        :returns: Nearest neighbor index or None if no session exists
        :rtype: LSHNearestNeighborIndex|None
        """
        session = ModelImporter.model('item').findOne({'_id': ObjectId(sid)})

        if not session:
            return None
        else:
            smqtkFolder = {'_id': ObjectId(session['meta']['smqtk_folder_id'])}

            functor = ItqFunctor(smqtkDataElementFromGirderFileId(
                                 localSmqtkFileIdFromName(smqtkFolder, 'mean_vec.npy')),
                                 smqtkDataElementFromGirderFileId(
                                 localSmqtkFileIdFromName(smqtkFolder, 'rotation.npy')))
            hash2uuidsKV = MemoryKeyValueStore(
                smqtkDataElementFromGirderFileId(localSmqtkFileIdFromName(smqtkFolder, 'hash2uuids.pickle')))

            return LSHNearestNeighborIndex(functor, descriptor_index,
                                           hash2uuidsKV, read_only=True)

    @access.user
    @filtermodel(model='item')
    @autoDescribeRoute(
        Description('Get an IQR session by ID.')
        .modelParam('id', model='item', level=AccessType.READ))
    def getSession(self, item, params):
        return item

    @access.user
    @autoDescribeRoute(
        Description('Create an IQR session.')
        .modelParam('smqtkFolder', model='folder', level=AccessType.READ, paramType='query'))
    def createSession(self, params):
        smqtkFolder = params['smqtkFolder']
        sessionsFolder = getCreateSessionsFolder()

        # Get the folder with images in it, since this is what's used for computing
        # what descriptor index table to use
        dataFolderId = ModelImporter.model('folder').load(ObjectId(smqtkFolder), user=getCurrentUser())
        dataFolderId = str(dataFolderId['parentId'])

        # Create session named after its id
        session = ModelImporter.model('item').createItem('placeholder_name',
                                                         getCurrentUser(), sessionsFolder)
        session['name'] = str(session['_id'])
        ModelImporter.model('item').save(session)
        sessionId = str(session['_id'])
        ModelImporter.model('item').setMetadata(session, {
            'smqtk_folder_id': smqtkFolder,
            'data_folder_id': dataFolderId,
            'pos_uuids': [],
            'neg_uuids': []
        })

        # already registered in the controller, return
        if self.controller.has_session_uuid(sessionId):
            return session

        iqrs = IqrSession(self.positive_seed_neighbors, session_uid=sessionId)

        with self.controller:
            with iqrs:  # because classifier maps locked by session
                self.controller.add_session(iqrs)
                self.session_classifiers[sessionId] = None
                self.session_classifier_dirty[sessionId] = True

        return session

    @access.user
    @autoDescribeRoute(
        Description('Refine adjudications of an IQR session.')
        .modelParam('id', model='item', level=AccessType.WRITE)
        .jsonParam('pos_uuids', '', required=True, paramType='body')
        .jsonParam('neg_uuids', '', required=False, paramType='body'))
    def refine(self, params):
        sid = str(params['item']['_id'])
        pos_uuids = params['pos_uuids']
        neg_uuids = params['neg_uuids'] if params['neg_uuids'] is not None else []

        if len(pos_uuids) == 0:
            raise RestException('No positive UUIDs given.')

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                raise RestException('Session ID %s not found.' % sid, 404)
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            descriptor_index = self._descriptorIndexFromSessionId(sid)
            neighbor_index = self._nearestNeighborIndex(sid, descriptor_index)

            if descriptor_index is None or neighbor_index is None:
                logger.error('Unable to compute descriptor or neighbor index from sid %s.' % sid)
                raise RestException('Unable to compute descriptor or neighbor index from sid %s.' % sid, 500)

            # Get appropriate descriptor elements from index for
            # setting new adjudication state.
            try:
                pos_descrs = set(descriptor_index.get_many_descriptors(pos_uuids))
                neg_descrs = set(descriptor_index.get_many_descriptors(neg_uuids))
            except KeyError as ex:
                logger.warn(traceback.format_exc())
                raise RestException('Descriptor UUID %s not found in index.' % ex, 404)

            # if a new classifier should be made upon the next
            # classification request.
            diff_pos = pos_descrs.symmetric_difference(iqrs.positive_descriptors)
            diff_neg = neg_descrs.symmetric_difference(iqrs.negative_descriptors)

            if diff_pos or diff_neg:
                logger.debug("[%s] session Classifier dirty", sid)
                self.session_classifier_dirty[sid] = True

            logger.info("[%s] Setting adjudications", sid)
            iqrs.positive_descriptors = pos_descrs
            iqrs.negative_descriptors = neg_descrs

            logger.info("[%s] Updating working index", sid)
            iqrs.update_working_index(neighbor_index)

            logger.info("[%s] Refining", sid)
            iqrs.refine()

        finally:
            iqrs.lock.release()

        return sid

    @access.user
    @filtermodel('item')
    @autoDescribeRoute(
        Description('Retrieve results from an IQR session.')
        .modelParam('sid', model='item', level=AccessType.WRITE, paramType='query')
        .param('limit', 'Maximum number of results to return.')
        .param('offset', 'Offset to start looking for results, useful for paginating results.'))
    def results(self, params):
        sid = params['sid']
        limit = int(params.get('limit', 25))

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                raise RestException('Session ID %s not found.' % sid, 404)
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            num_results = (iqrs.results and len(iqrs.results)) or 0

            offset = int(params.get('offset', num_results))

            uuid_dist = {}
            if iqrs.results:
                results = iqrs.ordered_results()[offset:limit]

                for descriptor, confidence in results:
                    uuid_dist[descriptor.uuid()] = confidence

        finally:
            iqrs.lock.release()

        dataFolder = {'_id': ObjectId(params['item']['meta']['data_folder_id'])}
        items = list(ModelImporter.model('folder').childItems(dataFolder, filters={'meta.smqtk_uuid': {
            '$in': uuid_dist.keys()
        }}))

        for item in items:
            item['meta']['smqtk_iqr_confidence'] = uuid_dist[item['meta']['smqtk_uuid']]

        return items
