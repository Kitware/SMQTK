from girder.utility.model_importer import ModelImporter
from girder.api.rest import getCurrentUser, filtermodel

from girder_client import HttpError


def getCreateFolder(gc, parentFolderId, name):
    try:
        # create/reuse existing
        smqtkFolder = gc.createFolder(parentFolderId, name)
    except HttpError:
        smqtkFolder = gc.get('folder', parameters={'parentId': parentFolderId,
                                                   'parentType': 'folder',
                                                   'name': name})[0]

    return smqtkFolder


@filtermodel(model='folder')
def getCreateSessionsFolder():
    user = getCurrentUser()
    folder = ModelImporter.model('folder')

    # @todo Assumes a Private folder will always exist/be accessible
    privateFolder = list(folder.childFolders(parentType='user',
                                             parent=user,
                                             user=user,
                                             filters={
                                                 'name': 'Private'
                                             }))[0]

    return folder.createFolder(privateFolder, 'iqr_sessions', reuseExisting=True)


def localSmqtkFileIdFromName(smqtkFolder, name):
    folder = ModelImporter.model('folder')
    item = list(folder.childItems(smqtkFolder, filters={'name': name}))[0]
    return str(list(ModelImporter.model('item').childFiles(item, limit=1))[0]['_id'])


def smqtkFileIdFromName(gc, smqtkFolder, name):
    item = list(gc.listItem(smqtkFolder['_id'], name=name))[0]
    return list(gc.listFile(item['_id']))[0]['_id']
