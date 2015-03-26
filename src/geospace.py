import datetime
import dateutil.parser
import json
import pymongo
# import urllib

import girder.api.rest
from girder import logger
from girder.api import access
from girder.api.describe import Description

class Scraper(girder.api.rest.Resource):
    def __init__(self):
        self.resourceName = 'scrape'
        self.route('GET', (), self.scrape)

    @access.public
    def scrape(self, params):
        result = []
        url = params.get('url', None)
        if url:
            import requests
            import BeautifulSoup as bs4
            print type(url)
            page = requests.get(url)
            print page.text
            #tree = html.fromstring(page.text)
            soup = bs4.BeautifulSoup(page.text)
            result = soup.findAll('img')
            result = [a.get('src') for a in result]
            print result
        return result

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

        # Get location if provided
        location = params.get('location', None)

        print 'location is ', location

        if location is not None:
            # Reverse geocode location
            print 'location is ', location

            from geopy.geocoders import Nominatim
            geolocator = Nominatim()
            location = geolocator.geocode(location)

            print 'location is ', location.longitude

            # Hard-coded to 1 unit for now
            geospatial_query = { "$geoWithin" : {
                    "$center" : [[location.longitude, location.latitude], 1],
                }
            }

        # Find all of the datasets within time range.
        time_range = eval(time_range)

        start = time_range[0]
        end = time_range[1]

        start_time = datetime.datetime.fromtimestamp(start).isoformat()
        end_time = datetime.datetime.fromtimestamp(end).isoformat()

        if location is None:
            query_result = coll.find({"field4":{"$gte": start_time, "$lt": end_time}}, skip=offset, limit=limit, sort=sort)
        else:
            query_result = coll.find({"field4":[{"$gte": start_time, "$lt": end_time}, geospatial_query]}, skip=offset, limit=limit, sort=sort)
        result = [row for row in query_result]
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
    info['apiRoot'].scrape = Scraper()