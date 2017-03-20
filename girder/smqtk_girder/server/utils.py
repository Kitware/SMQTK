from girder.utility.model_importer import ModelImporter
from girder.api.rest import getCurrentUser, filtermodel

import requests

import tempfile
import os

from smqtk.utils.image_utils import is_valid_element
from smqtk.utils import parallel
from smqtk.representation.data_element.file_element import DataFileElement

from urlparse import urlparse
from girder_client import GirderClient, HttpError

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



def smqtkFileIdFromName(gc, smqtkFolder, name):
    item = list(gc.listItem(smqtkFolder['_id'], name=name))[0]
    return list(gc.listFile(item['_id']))[0]['_id']
