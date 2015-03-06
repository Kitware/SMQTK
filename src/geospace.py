import datetime
import dateutil.parser
import json
import pymongo
# import urllib

import girder.api.rest
from girder import logger
from girder.api import access
from girder.api.describe import Description

class Geospace(girder.api.rest.Resource):
    """API endpoint for Geospace data."""

    def __init__(self):
        self.resourceName = 'data'
        self.route('GET', (), self.find)

    @access.public
    def find(self, params):
        limit, offset, sort = self.getPagingParameters(params)
        result = {}

        db = pymongo.MongoClient('mongodb://localhost:27017/ist')
        database = db.get_default_database()
        coll = database['ads']

        # Get valid time range
        time_range = params.get('duration', None)
        if (time_range is None):
            result['duration'] = {
                "start": [i for i in coll.find({"field4": { "$exists": True, "$nin": ["None"] } } ).sort("field4",  1).limit(1)][0],
                "end":   [i for i in coll.find({"field4": { "$exists": True, "$nin": ["None"] } } ).sort("field4", -1).limit(1)][0]
            };
            return result;

        # Find all of the datasets within time range.

        db = pymongo.MongoClient('mongodb://localhost:27017/ist')
        database = db.get_default_database()
        coll = database['ads']
        spec = coll.find(spec={}, skip=offset, limit=limit, sort=sort)
        result = [row for row in spec]
        return result

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

def load(info):
    info['apiRoot'].data = Geospace()