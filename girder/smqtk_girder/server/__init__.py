from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import filtermodel, getCurrentUser, Resource
from girder.constants import AccessType, TokenScope, SortDir
from girder.utility import setting_utilities
from girder.utility.model_importer import ModelImporter
from girder.plugins.worker.utils import getWorkerApiUrl, jobInfoSpec

from girder import logger

from .constants import PluginSettings

import itertools

SMQTK_SETTING_READ = 'smqtk_girder.setting_read'

@setting_utilities.validator({
    PluginSettings.DB_HOST,
    PluginSettings.DB_NAME,
    PluginSettings.DB_USER,
    PluginSettings.DB_PASS,
    PluginSettings.DB_DESCRIPTORS_TABLE,
    PluginSettings.IMAGE_BATCH_SIZE,
    PluginSettings.CAFFE_NETWORK_MODEL,
    PluginSettings.CAFFE_NETWORK_PROTOTXT,
    PluginSettings.CAFFE_IMAGE_MEAN
})
def validateSettings(doc):
    pass

@setting_utilities.default(PluginSettings.DB_HOST)
def defaultDbHost():
    return 'localhost'

@setting_utilities.default(PluginSettings.DB_NAME)
def defaultDbName():
    return 'smqtk'

@setting_utilities.default(PluginSettings.DB_USER)
def defaultDbUser():
    return 'smqtk'

@setting_utilities.default(PluginSettings.DB_DESCRIPTORS_TABLE)
def defaultDbDescriptorsTable():
    return 'descriptors'

@setting_utilities.default(PluginSettings.IMAGE_BATCH_SIZE)
def defaultImageBatchSize():
    return 100


class SmqtkAPI(Resource):
    def __init__(self):
        self.resourceName = 'smqtk'
        self.route('GET', ('settings',), self.settings)
        self.route('POST', ('process_images',), self.processImages)

        TokenScope.describeScope(SMQTK_SETTING_READ, 'Read SMQTK settings',
                                 'Allow clients to look up the SMQTK settings, including private '
                                 'fields such as database credentials.')

    @access.user(scope=SMQTK_SETTING_READ)
    @autoDescribeRoute(Description('Retrieve settings related to SMQTK.'))
    def settings(self, params):
        """
        Retrieve the settings related to SMQTK.

        NOTE: This returns *very sensitive* information including database
        credentials. This endpoint is intended to only be called from remote workers
        which has been registered with the task queue with their respective credentials.

        The only place tokens with this scope are assigned is within _processImages.
        """
        setting_results = list(ModelImporter.model('setting').find({
            'key': {
                '$regex': '^smqtk_girder\.'
            }
        }))

        return dict([(x['key'].replace('smqtk_girder.', ''), x['value']) for x in setting_results])

    @staticmethod
    def _processImages(folder, itemFilePairs):
        """
        Create and schedule a Girder job for processing images.

        :param folder: Folder to process images for.
        :param fileIds: File IDs to process, these are converted into Girder data elemnts.
        """
        jobModel = ModelImporter.model('job', 'jobs')

        # TODO Use a more granular token.
        # Ideally this would be scoped to only allow:
        # - Job Updates
        # - Data management of folder
        # - Retrieval of SMQTK settings
        token = ModelImporter.model('token').createToken(user=getCurrentUser(),
                                                         days=1,
                                                         scope=(TokenScope.USER_AUTH,
                                                                SMQTK_SETTING_READ))

        dataElementUris = [(itemId, 'girder://token:%s@%s/file/%s' % (token['_id'],
                                                                      getWorkerApiUrl(),
                                                                      fileId))
                           for (itemId, fileId) in itemFilePairs]

        job = jobModel.createJob(title='Processing Images',
                                 type='GPU',
                                 handler='worker_handler',
                                 user=getCurrentUser(),
                                 args=(str(folder['_id']), dataElementUris),
                                 otherFields={'celeryTaskName': 'smqtk_worker.tasks.process_images',
                                              'celeryQueue': 'process-images'})

        job['token'] = token

        logger.info('assigning token %s' % token['_id'])

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

        :param folder: A folder to process images on.
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
                return (str(item['_id']), files[0]['_id'])
            except Exception:
                return (False, False)

        # TODO Filter items by supported mime types for SMQTK
        items = itertools.ifilter(lambda item: 'smqtk_uuid' not in item.get('meta', {}),
                                  ModelImporter.model('folder').childItems(folder))

        self._processImages(folder, itertools.ifilter(None,
                                                      itertools.imap(oldestFileId, items)))


def load(info):
    from .nearest_neighbors import NearestNeighbors
    from .iqr import Iqr
    info['apiRoot'].smqtk_nearest_neighbors = NearestNeighbors()
    info['apiRoot'].smqtk = SmqtkAPI()
    info['apiRoot'].smqtk_iqr = Iqr()
