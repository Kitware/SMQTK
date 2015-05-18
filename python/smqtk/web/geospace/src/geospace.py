import datetime
import dateutil.parser
import json
import pymongo

import girder.api.rest
from girder import logger
from girder.api import access
from girder.api.describe import Description
import girder.models

# ----------------------------------------------------------------------------
class Scraper(girder.api.rest.Resource):
    """ Scrap a URL for images """
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
            page = requests.get(url)
            soup = bs4.BeautifulSoup(page.text)
            result = soup.findAll('img')
            result = [a.get('src') for a in result]
        return result

# ----------------------------------------------------------------------------
class Geospace(girder.api.rest.Resource):
    """API endpoint for Geospace data."""

    def __init__(self):
        self.resourceName = 'data'
        self.route('GET', (), self.find)

    @access.public
    def find(self, params):
        limit, offset, sort = self.getPagingParameters(params)
        result = {}
        loc = None
        search_radius = 1

        # TODO read it from the config file
        db = pymongo.MongoClient(girder.models.getDbConfig().get('isuri', 'mongodb://localhost:27017/ist'))
        database = db.get_default_database()
        coll = database['ads2']

        # Get valid time range
        time_range = params.get('duration', None)
        if (time_range is None):
            result['duration'] = {
                "start": [i for i in coll.find(
                    {"time": { "$exists": True, "$nin": ["None"] } } ).sort(
                        "time",  1).limit(1)][0],
                "end":   [i for i in coll.find(
                    {"time": { "$exists": True, "$nin": ["None"] } } ).sort(
                        "time", -1).limit(1)][0]
            };

            print result
            return result;

        # Get location if provided
        location = params.get('location', None)
        location_type = params.get('location_type', None)
        geospatial_query = None
        use_location = False

        if (location is not None) and (len(location) > 0):
            location = location.strip()
            if (location != "*"):
                use_location = True

                # Geocode location
                if (location_type == "address"):
                    from geopy.geocoders import Nominatim
                    geolocator = Nominatim()
                    loc = geolocator.geocode(location)
                    loc = {"latitude":loc.latitude, "longitude": loc.longitude};
                else:
                    try:
                        loc = location.split(",")
                        loc = {"latitude":float(loc[0]), "longitude": float(loc[1])};
                        search_radius = 0.1
                    except:
                        use_location = False
                        pass

                # Hard-coded to 10 units for now
                geospatial_query = {   "loc" : {"$geoWithin" :
                        {
                            "$center" : [[loc["latitude"], loc["longitude"]], search_radius]
                        }
                    }
                }

        # Find all of the datasets within time range.
        time_range = eval(time_range)

        start = time_range[0]
        end = time_range[1]

        start_time = start
        end_time = end

        if not use_location:
            query_result = coll.find({"time":
                {"$gte": start_time, "$lt": end_time}},
                skip=offset, limit=limit, sort=sort)
        else:
            # When using location do not limit by time for now
            query_result = coll.find( {"$and":[
                    geospatial_query,
                    {"time":
                        {"$gte": start_time, "$lt": end_time}
                    }]
                },
                skip=offset, limit=limit, sort=sort)
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

# ----------------------------------------------------------------------------
def load(info):
    info['apiRoot'].data = Geospace()
    info['apiRoot'].scrape = Scraper()
