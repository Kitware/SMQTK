"""
Utilities for interacting with Girder
"""

import datetime

import requests

from smqtk.utils import SmqtkObject
from smqtk.utils.url import url_join


class GirderTokenManager (SmqtkObject):
    """
    Helper class to manage storage and update of a Girder authentication token.

    This is better than using the girder_client module in that this class will
    request a new token with the current one expires.
    """

    def __init__(self, api_root='http://localhost:8080/api/v1', api_key=None):
        """
        Initialize a new token manager

        :param api_root: Girder API root URL
        :type api_root: str

        :param api_key: Optional API key to request token with. Otherwise an
        :type api_key:

        """
        self._api_root = api_root
        self._api_key = api_key
        self._token = None
        self._expiration = None

    @property
    def api_root(self):
        return self._api_root

    @property
    def api_key(self):
        return self._api_key

    @property
    def token(self):
        return self._token

    @property
    def expiration(self):
        return self._expiration

    @staticmethod
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
        if self._api_key:
            r = requests.post(
                url_join(self._api_root, 'api_key/token'),
                data={'key': self._api_key}
            )
            r.raise_for_status()
            token = r.json()['authToken']['token']
            expires = r.json()['authToken']['expires']
        else:
            r = requests.get(
                url_join(self._api_root, 'token/session')
            )
            r.raise_for_status()
            token = r.json()['token']
            expires = r.json()['expires']
        return token, self._parse_expiration_timestamp(expires)

    def _check_token_expiration(self):
        """
        Check if our current auth token has expired or if we don't have one yet.
        If so, request a new one. Only does anything if we have an api_key set.

        Assumes ``'self.api_key`` is defined and a valid API key value.

        :raises AssertionError: Expiration timestamp did not have a UTC timezone
            specifier attacked to the end.
        """
        if (self._token is None
                or self.token_expiration <= datetime.datetime.now()):
            self._log.debug("No or expired token")
            self._token, self.token_expiration = self._request_token()

    def get_token(self):
        self._check_token_expiration()
        return self._token

    def get_requests_header(self):
        """
        :return: The token authorization header if we have an api_key set.
            Otherwise returns None.
        :rtype: None | dict
        """
        self._check_token_expiration()
        return {'Girder-Token': self._token}
