import datetime
import dateutil.parser
import json
import pymongo
# import urllib

import girder.api.rest
from girder import logger
from girder.api import access
from girder.api.describe import Description

class Taxi(girder.api.rest.Resource):
    """API endpoint for taxi data."""

    def __init__(self):
        self.resourceName = 'data'
        self.route('GET', (), self.find)

    @access.public
    def find(self, params):
        limit, offset, sort = self.getPagingParameters(params)
        # fields = None
        # if 'fields' in params:
        #     fields = params['fields'].replace(',', ' ').strip().split()
        #     if not len(fields):
        #         fields = None
        # access = self.access[params.get('source', 'mongo')]
        # if isinstance(access, tuple):
        #     access = access[0](**access[1])
        #     self.access[params.get('source', 'mongo')] = access
        db = pymongo.MongoClient('mongodb://localhost:27017/ist')
        database = db.get_default_database()
        census = database['ads']
        spec = census.find(spec={}, skip=offset, limit=limit, sort=sort)
        result = [row for row in spec]

        # result['limit'] = limit
        # result['offset'] = offset
        # result['sort'] = sort
        # result['datacount'] = len(result.get('data', []))
        # if params.get('format', None) == 'list':
        #     if result.get('format', '') != 'list':
        #         if not fields:
        #             fields = FieldTable.keys()
        #         result['fields'] = fields
        #         result['columns'] = {fields[col]: col
        #                              for col in xrange(len(fields))}
        #         if 'data' in result:
        #             result['data'] = [
        #                 [row.get(field, None) for field in fields]
        #                 for row in result['data']
        #             ]
        #         result['format'] = 'list'
        # else:
        #     if result.get('format', '') == 'list':
        #         if 'data' in result:
        #             result['data'] = [{
        #                 result['fields'][col]: row[col]
        #                 for col in xrange(len(row))} for row in result['data']]
        #         result['format'] = 'dict'
        #         del result['columns']
        # We could let Girder convert the results into JSON, but it is
        # marginally faster to dump the JSON ourselves, since we can exclude
        # sorting and reduce whitespace
        return result

        # def resultFunc():
        #     yield json.dumps(
        #         result, check_circular=False, separators=(',', ':'),
        #         sort_keys=False, default=str)

        # cherrypy.response.headers['Content-Type'] = 'application/json'
        # return resultFunc

    find.description = (
        Description('Get a set of data.')
        .param('source', 'Database source (default mongo).', required=False,
               enum=['mongo', 'mongofull', 'tangelo'])
        .param('limit', 'Result set size limit (default=50).', required=False,
               dataType='int')
        .param('offset', 'Offset into result set (default=0).', required=False,
               dataType='int')
        .param('sort', 'Field to sort the user list by (default='
               'pickup_datetime)', required=False)
        .param('sortdir', '1 for ascending, -1 for descending (default=1)',
               required=False, dataType='int'))
        # .param('fields', 'A comma-separated list of fields to return (default '
        #        'is all fields).', required=False)
        # .param('format', 'The format to return the data (default is dict).',
        #        required=False, enum=['dict', 'list']))

    # for field in sorted(FieldTable):
    #     (fieldType, fieldDesc) = FieldTable[field]
    #     dataType = fieldType
    #     if dataType == 'text':
    #         dataType = 'string'
    #     find.description.param(field, fieldDesc, required=False,
    #                            dataType=dataType)
    #     if fieldType != 'text':
    #         find.description.param(
    #             field+'_min', 'Minimum value (inclusive) of ' + fieldDesc,
    #             required=False, dataType=dataType)
    #         find.description.param(
    #             field+'_max', 'Maximum value (exclusive) of ' + fieldDesc,
    #             required=False, dataType=dataType)

def load(info):
    info['apiRoot'].data = Taxi()
    # info['apiRoot'].collection.route('GET', (':id', 'quota'),
    #                                  quota.getCollectionQuota)
    # info['apiRoot'].collection.route('PUT', (':id', 'quota'),
    #                                  quota.setCollectionQuota)
    # info['apiRoot'].user.route('GET', (':id', 'quota'), quota.getUserQuota)
    # info['apiRoot'].user.route('PUT', (':id', 'quota'), quota.setUserQuota)
    # events.bind('model.setting.validate', 'userQuota', validateSettings)
    # events.bind('model.upload.assetstore', 'userQuota',
    #             quota.getUploadAssetstore)
    # events.bind('model.upload.save', 'userQuota', quota.checkUploadStart)
    # events.bind('model.upload.finalize', 'userQuota',
    #             quota.checkUploadFinalize)
