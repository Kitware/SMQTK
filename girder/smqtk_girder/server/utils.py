from bson.objectid import ObjectId

from smqtk.representation.data_element.memory_element import DataMemoryElement

from girder.utility.model_importer import ModelImporter
from girder.api.rest import getCurrentUser, filtermodel

from girder_client import HttpError


def getCreateFolder(gc, parentFolderId, name):
    """
    Gets or creates a folder on the remote host with the given name.

    :param gc: Authenticated GirderClient instance
    :param parentFolderId: The ID of the parent folder
    :param name: Name of the folder to be created/retrieved
    :returns: The created (or already existing) folder.
    :rtype: dict
    """
    try:
        # create/reuse existing
        # note: this function can be simplified once
        # https://github.com/girder/girder/pull/2072 is merged.
        smqtkFolder = gc.createFolder(parentFolderId, name)
    except HttpError:
        smqtkFolder = gc.get('folder', parameters={'parentId': parentFolderId,
                                                   'parentType': 'folder',
                                                   'name': name})[0]

    return smqtkFolder


@filtermodel(model='folder')
def getCreateSessionsFolder():
    """
    Gets or creates a folder for storing IQR sessions

    :returns: The created (or already existing) folder
    :rtype: dict
    """
    user = getCurrentUser()
    folder = ModelImporter.model('folder')
    return folder.createFolder(user, '.smqtk_iqr_sessions', parentType='user', reuseExisting=True)


def smqtkDataElementFromGirderFileId(fileId):
    file = ModelImporter.model('file').load(ObjectId(fileId), force=True)
    with ModelImporter.model('file').open(file) as fh:
        return DataMemoryElement(bytes=fh.read(file['size']), readonly=True)


def localSmqtkFileIdFromName(smqtkFolder, name):
    folder = ModelImporter.model('folder')
    item = list(folder.childItems(smqtkFolder, filters={'name': name}))[0]
    return str(list(ModelImporter.model('item').childFiles(item, limit=1))[0]['_id'])


def smqtkFileIdFromName(gc, smqtkFolder, name):
    """
    Retrieve a SMQTK file ID from a given name.

    SMQTK file means a file stored in the .smqtk folder given.

    :param gc: An authenticated GirderClient instance
    :param smqtkFolder: The .smqtk folder object
    :param name: Name of the file
    :returns: File ID
    """
    item = list(gc.listItem(smqtkFolder['_id'], name=name))[0]
    return list(gc.listFile(item['_id']))[0]['_id']
