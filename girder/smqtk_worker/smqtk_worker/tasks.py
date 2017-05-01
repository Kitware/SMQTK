import celery
import cPickle as pickle
import six

from time import sleep
from smqtk import compute_functions
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.girder import GirderDataElement
from smqtk.representation.descriptor_element.postgres import PostgresDescriptorElement
from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.algorithms.descriptor_generator.caffe_descriptor import CaffeDescriptorGenerator
from girder_worker.app import app
from .utils import (iter_valid_elements, getCreateFolder, createOverwriteItem,
                    initializeItemWithFile, smqtkFileIdFromName, descriptorIndexFromFolderId,
                    getSetting)


def girderUriFromTask(task, fileId):
    return 'girder://token:%s@%s/file/%s' % (task.request.jobInfoSpec['headers']['Girder-Token'],
                                             task.request.apiUrl,
                                             fileId)


@app.task(bind=True, queue='compute-descriptors')
def compute_descriptors(task, folderId, dataElementUris, **kwargs):
    """
    Celery task for computing descriptors for a series of data element URIs
    belonging to a single folder.

    After computing descriptors for a series of Girder files, the relevant items
    are updated within Girder to contain the smqtk_uuid (sha1) value as metadata.

    :param task: Celery provided task object.
    :param folderId: The folder these images are related to, this is used for
        namespacing the descriptor index table.
    :param dataElementUris: A list of data element URIs, these are assumed to be
        GirderDataElement URIs.
    """
    task.job_manager.updateProgress(message='Computing descriptors', forceFlush=True)
    generator = CaffeDescriptorGenerator(
        girderUriFromTask(task, getSetting(task.girder_client, 'network_prototxt_fileid')),
        girderUriFromTask(task, getSetting(task.girder_client, 'network_model_fileid')),
        girderUriFromTask(task, getSetting(task.girder_client, 'image_mean_fileid')))

    factory = DescriptorElementFactory(PostgresDescriptorElement, {
        'db_name': getSetting(task.girder_client, 'db_name'),
        'db_host': getSetting(task.girder_client, 'db_host'),
        'db_user': getSetting(task.girder_client, 'db_user'),
        'db_pass': getSetting(task.girder_client, 'db_pass')
    })

    index = descriptorIndexFromFolderId(task.girder_client, folderId)

    valid_elements = iter_valid_elements(dataElementUris, generator.valid_content_types())

    descriptors = compute_functions.compute_many_descriptors(valid_elements,
                                                             generator,
                                                             factory,
                                                             index,
                                                             use_mp=False)

    for de, descriptor in descriptors:
        # TODO Catch errors that could occur here
        # TODO Batch requests
        task.girder_client.addMetadataToItem(
            task.girder_client.get('file/%s' % de.file_id)['itemId'], {
                'smqtk_uuid': descriptor.uuid()
        })


@app.task(bind=True, queue='compute-descriptors')
def itq(task, folderId, **kwargs):
    """
    Celery task for training ITQ on a given folder.

    This trains ITQ on all descriptors within the index. Since this
    is typically called after computing descriptors, it will often
    only contain what's in the folder.

    :param task: Celery provided task object.
    :param folderId: The folder to train ITQ for, note this is only used to
        infer the descriptor index.
    """
    task.job_manager.updateProgress(message='Training ITQ', forceFlush=True)
    index = descriptorIndexFromFolderId(task.girder_client, folderId)

    if not index.count():
        # TODO SMQTK should account for this?
        raise Exception('Descriptor index is empty, cannot train ITQ.')

    smqtkFolder = getCreateFolder(task.girder_client, folderId, '.smqtk')
    meanVecFile = initializeItemWithFile(task.girder_client,
                                         createOverwriteItem(task.girder_client, smqtkFolder['_id'], 'mean_vec.npy'))
    rotationFile = initializeItemWithFile(task.girder_client,
                                          createOverwriteItem(task.girder_client, smqtkFolder['_id'], 'rotation.npy'))

    functor = ItqFunctor(mean_vec_cache=GirderDataElement(meanVecFile['_id'], api_root=task.request.apiUrl,
                                                          token=task.request.jobInfoSpec['headers']['Girder-Token']),
                         rotation_cache=GirderDataElement(rotationFile['_id'], api_root=task.request.apiUrl,
                                                          token=task.request.jobInfoSpec['headers']['Girder-Token']))

    functor.fit(index.iterdescriptors(), use_multiprocessing=False)


@app.task(bind=True, queue='compute-descriptors')
def compute_hash_codes(task, folderId, **kwargs):
    """
    Celery task for computing hash codes on a given folder (descriptor index).

    :param task: Celery provided task object.
    :param folderId: The folder to train ITQ for, note this is only used to
        infer the descriptor index.
    """
    task.job_manager.updateProgress(message='Computing Hash Codes', forceFlush=True)

    index = descriptorIndexFromFolderId(task.girder_client, folderId)

    smqtkFolder = getCreateFolder(task.girder_client, folderId, '.smqtk')

    meanVecFileId = smqtkFileIdFromName(task.girder_client, smqtkFolder, 'mean_vec.npy')
    rotationFileId = smqtkFileIdFromName(task.girder_client, smqtkFolder, 'rotation.npy')
    hash2uuidsFile = initializeItemWithFile(task.girder_client,
                                            createOverwriteItem(task.girder_client, smqtkFolder['_id'], 'hash2uuids.pickle'))

    functor = ItqFunctor(mean_vec_cache=GirderDataElement(meanVecFileId, api_root=task.request.apiUrl,
                                                          token=task.request.jobInfoSpec['headers']['Girder-Token']),
                         rotation_cache=GirderDataElement(rotationFileId, api_root=task.request.apiUrl,
                                                          token=task.request.jobInfoSpec['headers']['Girder-Token']))

    hash2uuids = compute_functions.compute_hash_codes(index.iterkeys(), index, functor, use_mp=False)

    data = pickle.dumps(hash2uuids)
    task.girder_client.uploadFileContents(hash2uuidsFile['_id'], six.BytesIO(data), len(data))


@app.task(bind=True)
def process_images(task, folderId, dataElementUris, **kwargs):
    """
    Celery task for processing images so they can be queried through
    nearest neighbors and IQR services.

    This performs a "chain" of tasks over the dataElementUris given.

    If dataElementUris is empty, only training ITQ and computing hash codes
    will be called. Otherwise, descriptor computation will be performed first.

    Note this task must be synchronous to ensure that Girder progress updates are
    accurate.

    :param task: Celery provided task object.
    :param folderId: The folderId to process images from, this is used for creating
        the descriptor index for that folder.
    :param dataElementUris: A list of Girder data element URIs.
    """
    task.job_manager.updateProgress(message='Processing Images', forceFlush=True)
    # Each compute descriptors job ensures this table exists
    # but we run into a weird issue when they're running at the same time, see
    # https://www.postgresql.org/message-id/4B967376.7050300%40opinioni.net
    # To work around it, create the table before mapping the jobs.
    index = descriptorIndexFromFolderId(task.girder_client, folderId)
    conn = index._get_psql_connection()
    with conn:
        with conn.cursor() as cur:
            index._ensure_table(cur)

    batch_size = int(getSetting(task.girder_client, 'process_images_batch_size'))
    batches = [dataElementUris[x:x+batch_size]
               for x in xrange(0, len(dataElementUris), batch_size)]

    # The grouped task (descriptor_jobs) is calling compute_descriptors in a batched
    # fashion so that it can be distributed across many machines. The next parts of
    # the chain perform ITQ and CHC. These parts are marked immutable so their
    # arguments can't be changed. That's because these jobs don't care about the output
    # of the descriptor computation jobs.
    descriptor_jobs = [compute_descriptors.signature((folderId, batch), {},
                                                     headers={'jobInfoSpec': task.request.jobInfoSpec,
                                                              'apiUrl': task.request.apiUrl}) for batch in batches]
    itq_job = itq.signature((folderId,), {}, headers={'jobInfoSpec': task.request.jobInfoSpec,
                                                      'apiUrl': task.request.apiUrl},
                            queue='compute-descriptors', immutable=True)
    chc_job = compute_hash_codes.signature((folderId,), {}, headers={'jobInfoSpec': task.request.jobInfoSpec,
                                                                     'apiUrl': task.request.apiUrl},
                                           queue='compute-descriptors', immutable=True)

    # It's possible all of the descriptors have already been computed for this dir
    if descriptor_jobs:
        result = (celery.group(descriptor_jobs) | itq_job | chc_job).apply_async(throw=True)
    else:
        result = (itq_job | chc_job).delay()

    # Typically this is done using result.get(), however that can result in deadlocking due to
    # how Celery workers prefetch jobs. As a result it throws an exception. The way we've arranged
    # the queues (compute-descriptors and process-images), the deadlocking won't occur - but Celery
    # still raises a warning/exception. So we manually perform the work of result.get() below.
    while True:
        if result.ready():
            if result.failed():
                result.throw()
            break
        sleep(.1)
