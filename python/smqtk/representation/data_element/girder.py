import datetime
import requests

from smqtk.representation import DataElement
from smqtk.utils.url import url_join


__all__ = [
    'GirderDataElement',
]


class GirderDataElement (DataElement):
    """
    Element whose data is stored via a Girder backend.  Accesses via Girder
    REST API given user credentials.
    """

    @classmethod
    def is_usable(cls):
        """
        Usable if we were able to import girder_client
        :return:
        :rtype:
        """
        # Requests module is a basic requirement
        return True

    def __init__(self, file_id, api_key=None,
                 girder_rest_root='http://localhost:8080/api/v1'):
        """
        Initialize data element to point to a specific file hosted in Girder

        An authorization token will be generated if an API key is provided at
        construction.  A new token will be requested if it has expired.

        :param file_id:
        :type file_id:
        :param api_key:
        :type api_key:
        :param girder_rest_root:
        :type girder_rest_root:
        """
        # TODO: Template sub-URLs for customizing model/download endpoints used?
        #       e.g. 'file/{file_id:s}' and '
        super(GirderDataElement, self).__init__()

        self.file_id = file_id
        self.api_key = api_key
        self.girder_rest_root = girder_rest_root

        self.token = None
        self.token_expiration = None

        # Cache so we don't have to query server multiple times for multiple
        # calls.
        self._content_type = None

    def _request_token(self):
        """
        Request an authentication token given an API key.

        Assumes ``'self.api_key`` is defined and a valid API key value.

        :raises AssertionError: Expiration timestamp did not have a UTC timezone
            specifier attacked to the end.

        :return: token string and expiration timestamp
        :rtype: str, datetime.datetime
        """
        self._log.debug("Requesting new authorization token.")
        r = requests.post(
            url_join(self.girder_rest_root, 'api_key/token'),
            data={'key': self.api_key}
        )
        _handle_error(r)
        token = r.json()['authToken']['token']
        expires = r.json()['authToken']['expires']
        return token, _parse_expiration_timestamp(expires)

    def _check_token_expiration(self):
        """
        Check if our current auth token has expired or if we don't have one yet.
        If so, request a new one. Only does anything if we have an api_key set.

        Assumes ``'self.api_key`` is defined and a valid API key value.

        :raises AssertionError: Expiration timestamp did not have a UTC timezone
            specifier attacked to the end.
        """
        if (self.token is None
                or self.token_expiration <= datetime.datetime.now()):
            self._log.debug("No or expired token")
            self.token, self.token_expiration = self._request_token()

    def _get_token_header(self):
        """
        :return: The token authorization header if we have an api_key set.
            Otherwise returns None.
        :rtype: None | dict
        """
        if self.api_key:
            self._check_token_expiration()
            return {'Girder-Token': self.token}
        return None

    def get_config(self):
        return {
            'file_id': self.file_id,
            'api_key': self.api_key,
            'girder_rest_root': self.girder_rest_root,
        }

    def content_type(self):
        # Check if token has expired, if so get new one
        # Get file model, which has mimetype info
        if self._content_type is None:
            self._log.debug("Getting content type for file %s", self.file_id)
            token_header = self._get_token_header()
            r = requests.get(url_join(self.girder_rest_root, 'file', self.file_id),
                             headers=token_header)
            _handle_error(r)
            self._content_type = r.json()['mimeType']
        return self._content_type

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes

        :raises AssertionError: Content received not the expected length in
            bytes (header field vs. content length).
        """
        # Check if token has expired, if so get new one
        # Download file bytes from girder
        self._log.debug("Getting bytes for file %s", self.file_id)
        token_header = self._get_token_header()
        r = requests.get(url_join(self.girder_rest_root, 'file', self.file_id,
                                  'download'),
                         params={'contentDisposition': 'inline'},
                         headers=token_header)
        _handle_error(r)
        content = r.content
        expected_length = int(r.headers['Content-Length'])
        assert len(content) == expected_length, \
            "Content received no the expected length: %d != %d (expected)" \
            % (len(content), expected_length)
        return content


def _parse_expiration_timestamp(ts):
    """
    Parse datetime instance from the given expiration timestamp string.
    Currently ignores timezone, but asserts that its '+00:00' (UTC).

    :raises AssertionError: Timestamp did not have a UTC timezone specifier
        attacked to the end.

    :param ts: Token expiration timestamp string.
    :type ts: unicode
    :return: Datetime instance parsed from timestamp.
    :rtype: datetime.datetime
    """
    # example:
    #   2017-02-25T14:59:48.333777+00:00
    # Ignoring timezone at the end for now and asserting that its always
    # '+00:00' (UTC)
    ts, tz = ts[:-6], ts[-6:]
    assert tz == "+00:00", "Expiration UTC timezone assumption broken, " \
                           "received: '%s'" % tz
    dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
    return dt


def _handle_error(r):
    """
    Handle if received an invalid response.

    :param r: Requests response
    :type r: requests.Response
    """
    r.raise_for_status()
