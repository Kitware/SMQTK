import time

import flask
import requests

from smqtk.utils.dict import merge_dict


def make_response_json(message, return_code=200, **params):
    """
    Basic message constructor for returning JSON from a flask routing function

    :param message: String descriptive message to send back.
    :type message: str

    :param return_code: HTTP return code for this message. Default is 200.
    :type return_code: int

    :param params: Other key-value data to include in response JSON.
    :type params: JSON-compliant

    :return: Flask response and HTTP status code pair.
    :rtype: (flask.Response, int)

    """
    r = {
        "message": message,
        "time": {
            "unix": time.time(),
            "utc": time.asctime(time.gmtime()),
        }
    }
    merge_dict(r, params)
    return flask.jsonify(**r), return_code


class ServiceProxy (object):
    """
    Helper class for interacting with an external service.
    """

    def __init__(self, url):
        """
        Parameters
        ---
            url : str
                URL to base requests on.
        """
        # Append http:// to the head of the URL if neither http(s) are present
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'http://' + url
        self.url = url

    def _compose(self, endpoint):
        return '/'.join([self.url, endpoint])

    def get(self, endpoint, **params):
        # Make params None if its empty.
        params = params and params or None
        return requests.get(self._compose(endpoint), params)

    def post(self, endpoint, **params):
        # Make params None if its empty.
        params = params and params or None
        return requests.post(self._compose(endpoint), data=params)

    def put(self, endpoint, **params):
        # Make params None if its empty.
        params = params and params or None
        return requests.put(self._compose(endpoint), data=params)

    def delete(self, endpoint, **params):
        # Make params None if its empty.
        params = params and params or None
        return requests.delete(self._compose(endpoint), params=params)
