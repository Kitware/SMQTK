import itertools

from smqtk.utils.image import is_valid_element
from smqtk.utils import parallel
from smqtk.representation.descriptor_set.postgres import PostgresDescriptorSet
from smqtk.representation.data_element.girder import GirderDataElement

from girder_client import HttpError

settings = None


def getSetting(gc, key=None):
    global settings

    if settings is None:
        settings = gc.get('smqtk/settings')

    if key is None:
        return settings
    else:
        return settings.get(key, None)


def descriptorSetFromFolderId(gc, folderId):
    return PostgresDescriptorSet('descriptor_set_%s' % folderId,
                                 db_name=getSetting(gc, 'db_name'),
                                 db_host=getSetting(gc, 'db_host'),
                                 db_user=getSetting(gc, 'db_user'),
                                 db_pass=getSetting(gc, 'db_pass'))


def smqtkFileIdFromName(gc, smqtkFolder, name):
    item = list(gc.listItem(smqtkFolder['_id'], name=name))[0]
    return list(gc.listFile(item['_id']))[0]['_id']


def getCreateFolder(gc, parentFolderId, name):
    try:
        # create/reuse existing
        smqtkFolder = gc.createFolder(parentFolderId, name)
    except HttpError:
        smqtkFolder = gc.get('folder', parameters={'parentId': parentFolderId,
                                                   'parentType': 'folder',
                                                   'name': name})[0]

    return smqtkFolder


def createOverwriteItem(gc, parentFolderId, name):
    """
    Creates an item, overwriting it if it already existed.

    :param gc: Instance of GirderClient with the correct permissions.
    :param parentFolderId: The parent folder of the item.
    :param name: The name of the item to create.
    :returns: The newly created item.
    :rtype: dict
    """
    toDelete = gc.listItem(parentFolderId, name=name)

    for item in toDelete:
        gc.delete('item/%s' % item['_id'])

    return gc.createItem(parentFolderId, name)


def initializeItemWithFile(gc, item):
    """
    Initializes an item with an empty file, returning that file.

    :param gc: Instance of GirderClient with the correct permissions.
    :param item: The item (dictionary) to initialize.
    :returns: The newly created file
    :rtype: dict
    """
    return gc.post('/file', {'parentId': item['_id'],
                             'parentType': 'item',
                             'size': 0,
                             'name': item['name']})


def iter_valid_elements(dataElementUris, valid_content_types):
    """
    Find the GirderDataElements which are loadable images and
    valid according to valid_content_types.

    :param dataElementUris: A list of Girder Data Element URIs.
    :param valid_content_types: A list of valid content types, generally
        passed by a descriptor generator.
    :returns: A generator over valid GirderDataElements.
    :rtype: generator
    """
    def is_valid(dataElementUri):
        dfe = GirderDataElement.from_uri(dataElementUri)

        if is_valid_element(dfe,
                            valid_content_types=valid_content_types,
                            check_image=True):
            return dfe
        else:
            return False

    return itertools.ifilter(None, parallel.parallel_map(is_valid,
                                                         dataElementUris,
                                                         use_multiprocessing=False))
