import requests

from smqtk.exceptions import ReadOnlyError
from smqtk.representation import DataElement
from smqtk.utils.girder import GirderTokenManager
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

    # TODO: from_uri
    #       - adapt http format for username/password specification
    #           (i.e. girder://<user>:<pass>@<url...>
    #       - maybe optionally allow API key in place of user/pass spec
    #           (i.e. girder://<api_key>@<url...>
    #       - <url> in above I guess would be the api/v1/... URL, including any
    #           parameters needed

    def __init__(self, file_id, api_root='http://localhost:8080/api/v1',
                 api_key=None):
        """
        Initialize data element to point to a specific file hosted in Girder

        An authorization token will be generated if an API key is provided at
        construction.  A new token will be requested if it has expired.

        :param file_id: ID of the file in Girder
        :type file_id: str

        :param api_root: Girder API root URL
        :type api_root: str

        :param api_key: Optional API key to request token with. Otherwise an
        :type api_key:

        """
        # TODO: Template sub-URLs for customizing model/download endpoints used?
        #       e.g. 'file/{file_id:s}' and '
        super(GirderDataElement, self).__init__()

        self.file_id = file_id
        self.api_root = api_root
        self.token_manager = GirderTokenManager(api_root, api_key)

        self.token = None
        self.token_expiration = None

        # Cache so we don't have to query server multiple times for multiple
        # calls.
        self._content_type = None

    def get_config(self):
        return {
            'file_id': self.file_id,
            'api_root': self.api_root,
            'api_key': self.token_manager._api_key,
        }

    def content_type(self):
        # Check if token has expired, if so get new one
        # Get file model, which has mimetype info
        if self._content_type is None:
            self._log.debug("Getting content type for file ID %s", self.file_id)
            token_header = self.token_manager.get_requests_header()
            r = requests.get(url_join(self.api_root, 'file', self.file_id),
                             headers=token_header)
            r.raise_for_status()
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
        self._log.debug("Getting bytes for file ID %s", self.file_id)
        token_header = self.token_manager.get_requests_header()
        r = requests.get(url_join(self.api_root, 'file', self.file_id,
                                  'download'),
                         params={'contentDisposition': 'inline'},
                         headers=token_header)
        r.raise_for_status()
        content = r.content
        expected_length = int(r.headers['Content-Length'])
        assert len(content) == expected_length, \
            "Content received no the expected length: %d != %d (expected)" \
            % (len(content), expected_length)
        return content

    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        # Current do not support writing to girder elements
        # TODO: Implement using PUT file/{id} endpoint if the file exists
        return False

    def set_bytes(self, b):
        """
        Set bytes to this data element in the form of a string.

        Not all implementations may support setting bytes (writing). See the
        ``writable`` method.

        :param b: bytes to set.
        :type b: str

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.

        """
        raise ReadOnlyError("Cannot write to Girder data elements.")
