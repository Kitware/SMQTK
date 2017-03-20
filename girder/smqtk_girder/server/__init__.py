from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import filtermodel, getCurrentUser, Resource
from girder.constants import AccessType, TokenScope, SortDir
from girder.utility.model_importer import ModelImporter
from girder.plugins.worker.utils import getWorkerApiUrl, jobInfoSpec

from girder import logger

import itertools


class SmqtkAPI(Resource):
    def __init__(self):
        self.resourceName = 'smqtk'
        self.route('POST', ('process_images',), self.processImages)

    @staticmethod
    def _processImages(folder, fileIds):
        """
        Create and schedule a Girder job for processing images.

        :param folder: Folder to process images for.
        :param fileIds: File IDs to process, these are converted into Girder data elemnts.
        """
        jobModel = ModelImporter.model('job', 'jobs')

        # TODO Use a more granular token.
        # Ideally this would be scoped to only allow job updates and data management of folder
        token = ModelImporter.model('token').createToken(user=getCurrentUser(),
                                                         days=1,
                                                         scope=TokenScope.USER_AUTH)

        dataElementUris = ['girder://token:%s@%s/file/%s' % (token['_id'],
                                                             getWorkerApiUrl(),
                                                             fileId)
                           for fileId in fileIds]

        job = jobModel.createJob(title='Processing Images',
                                 type='GPU',
                                 handler='worker_handler',
                                 user=getCurrentUser(),
                                 args=(str(folder['_id']), dataElementUris),
                                 otherFields={'celeryTaskName': 'smqtk_worker.tasks.process_images',
                                              'celeryQueue': 'process-images'})

        job['token'] = token


        logger.info('assigning token %s' % token['_id'])
        #job['kwargs']['jobInfo'] =

        jobModel.save(job)
        jobModel.scheduleJob(job)

    @access.user
    @filtermodel(model='folder')
    @autoDescribeRoute(Description('Compute descriptors on a given folder.')
                       .modelParam('id', model='folder', level=AccessType.READ))
    def processImages(self, folder, params):
        """
        Process the images of a directory. This includes computing descriptors
        as well as training ITQ and computing hash codes.

        :param folder:
        :param params:
        :returns:
        """
        def oldestFileId(item):
            """
            Find the oldest file in an item, and return its id.

            :param item: An item document, or minimally a dictionary with the item id.
            :returns: The id of the oldest file, or False if the item has no files.
            """
            files = ModelImporter.model('item').childFiles(item,
                                                           limit=1,
                                                           sort=[('created', SortDir.ASCENDING)])
            try:
                return files[0]['_id']
            except Exception:
                return False

        # TODO Filter items by supported mime types for SMQTK
        items = itertools.ifilter(lambda item: 'smqtk_uuid' not in item.get('meta', {}),
                                  ModelImporter.model('folder').childItems(folder))

        self._processImages(folder, itertools.ifilter(None,
                                                      itertools.imap(oldestFileId, items)))


def load(info):
    from .nearest_neighbors import NearestNeighbors
    info['apiRoot'].smqtk_nearest_neighbors = NearestNeighbors()
    info['apiRoot'].smqtk = SmqtkAPI()
