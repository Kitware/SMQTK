import time

import flask

from smqtk.utils import merge_dict


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
