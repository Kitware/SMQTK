from urlparse import urlparse
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
        :return: If this element type is usable
        :rtype: bool
        """
        # Requests module is a basic requirement
        # URLs are not necessarily on the public internet.
        return True

    # TODO: from_uri
    #       - maybe optionally allow API key in place of user/pass spec
    #           (i.e. girder://<api_key>@<url...>
    #       - <url> in above I guess would be the api/v1/... URL, including any
    #           parameters needed
    @classmethod
    def from_uri(cls, uri):
        """
        Creates a ``GirderDataElement`` from ``uri``.

        The parsing is accomplished by passing ``uri`` to ``urlparse``.
        This allows for quite a bit of flexibility in the types of URIs that
        can be passed. Valid Girder URIs are:

        (a) girder(s)://<user>:<pass>@<host>:<port>/api/v1/file/<file_id>
        (b) girder://<user>:<pass>@file:<file_id>

        <user> and <pass> are optional in both (a) and (b). <port> is optional
        in (a).

        If the (b) form of a Girder URI is used, then the ``api_root`` member
        will be the default of 'http://localhost:8080/api/v1'.

        Currently, only GirderDataElements can be built from Files, so the
        URL should end in /file/{id}.

        :param uri: A URI of the form
        girder(s)://<user>:<pass>@<host>:<port>/api/<version>/file/<file_id> or
        girder://<user>:<pass>@file:<file_id>.
        :type uri: str

        :return: Data element created from ``uri``
        :rtype: GirderDataElement

        :raises ValueError: An invalid URI was passed.

        :raises AssertionError: If the path parsed from a URI of the form (a)
        does not have '/file/' as its penultimate location before the
        identifier.
        """
        api_root = None
        # urlparse seems to not be phased by the 'girder' protocol instead of
        # http, so no replacing needs to be done.
        parsed_uri = urlparse(uri)
        if not parsed_uri.scheme.startswith('girder'):
            raise ValueError('Invalid Girder URI. Girder URIs must start with '
                             'girder:// or girders://')

        # For the API root to be valid, the URI must have a netloc
        # and the ``path`` must start with '/api'. This clause deals with
        # URIs of the form:
        # girder://<user>:<pass>@<host>:<port>/api/v1/file/<file_id>
        # The logic is constructed by understanding how urlparse parses a
        # URI of the above form. <port> is optional
        if parsed_uri.path.startswith('/api') and parsed_uri.netloc:
            assert parsed_uri.path.split('/file/')[0] != parsed_uri.path
            # If you're passing a URI of the form (a), either <girder> or
            # <girders> are valid tags. We determine the scheme to use in
            # constructing the api_root here based on that tag.
            if parsed_uri.scheme == 'girder':
                api_root = 'http://'
            elif parsed_uri.scheme == 'girders':
                api_root = 'https://'

            # We rsplit on '/file/' to get the preceding path information
            # before /file/<file_id>, which is used to construct the api root.
            api_root += '%s%s' % (parsed_uri.netloc,
                                  parsed_uri.path.rsplit('/file/')[0])
            file_id = parsed_uri.path.split('/')[-1]

        # This covers the case of a URI of the form:
        # girder://<user>:<pass>@file:<file_id>
        elif parsed_uri.netloc.startswith('file'):
            file_id = parsed_uri.netloc.split(':')[-1]

        # The above are the two currently supported forms of a Girder URI in
        # SMQTK. Anything else will be considered invalid.
        else:
            raise ValueError(
                'Invalid Girder URI. Girder URIs must be of the form: \n'
                '* girder://<user>:<pass>@<host>:<port>/api/<version>/'
                'file/<file_id>\n'
                '* girder://<user>:<pass>@file:<file_id>\n'
                'Where <user> and <pass> are optional.'
            )

        if api_root:
            girder_element = cls(file_id, api_root)
        else:
            girder_element = cls(file_id)
        return girder_element

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
        # TODO: Should token manager become a property so that if ``api_key``
        # is set after-the-fact the GirderTokenManager is instantiated
        # properly?
        self.token_manager = GirderTokenManager(api_root, api_key)

        self.token = None
        self.token_expiration = None

        # Cache so we don't have to query server multiple times for multiple
        # calls.
        self._content_type = None

    def __repr__(self):
        return super(GirderDataElement, self).__repr__() + \
            "{id: %s, api_root: %s, api_key: %s}" % (
                self.file_id, self.api_root, self.token_manager.api_key
            )

    def get_config(self):
        return {
            'file_id': self.file_id,
            'api_root': self.api_root,
            'api_key': self.token_manager.api_key,
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

    def get_file_model(self):
        """
        Get the file model json from the server.

        Returns None if the file does not exist on the server.

        :return: file model model as a dictionary
        :rtype: dict | None

        """
        r = requests.get(url_join(self.api_root, 'file', self.file_id),
                         headers=self.token_manager.get_requests_header())
        if r.status_code == 400:
            return None
        # Exception for any other status
        r.raise_for_status()
        return r.json()

    def is_empty(self):
        """
        Check if this element contains no bytes.

        This plugin checks that we can get a file model return from girder and
        that the size of the file queried is non-zero.

        :return: If there is a model for our item or if our item contains 0
            bytes.
        :rtype: bool

        """
        m = self.get_file_model()
        return m is None or m['size'] == 0

    def get_bytes(self):
        """
        Get the bytes of the file stored in girder.

        :return: Get the byte stream for this data element.
        :rtype: bytes

        :raises AssertionError: Content received not the expected length in
            bytes (header field vs. content length).
        :raises requests.HTTPError: If the ID does not refer to a file in
            Girder.
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
