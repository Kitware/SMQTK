import six
from six.moves.urllib_parse import urlparse

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation import DataElement

try:
    import girder_client
except ImportError:
    girder_client = None


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
        return girder_client is not None

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

        (a) girder://token:<token>@<api_url>/file/<file_id>
        (b) girder://api_key:<api_key>@<api_url>/file/<file_id>

        Currently, only GirderDataElements can be built from Files, so the
        URL should end in /file/{id}.

        :return: Data element created from ``uri``
        :rtype: GirderDataElement

        :raises ValueError: An invalid URI was passed.

        :raises AssertionError: If the path parsed from a URI of the form (a)
            does not have '/file/' as its penultimate location before the
            identifier.

        """
        # urlparse seems to not be phased by the 'girder' protocol instead of
        # http, so no replacing needs to be done.
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme != 'girder':
            raise InvalidUriError(uri, 'Invalid Girder URI. Girder URIs must '
                                       'start with girder://')

        if not parsed_uri.netloc:
            raise InvalidUriError(uri, 'No parsed netloc from given URI.')

        token = api_key = None

        if '@' in parsed_uri.netloc:
            credentials, scheme = parsed_uri.netloc.split('@')
            cred_type, cred = credentials.split(':')

            if cred_type == 'token':
                token = cred
            elif cred_type == 'api_key':
                api_key = cred
        else:
            scheme = parsed_uri.netloc

        try:
            path, file_id = parsed_uri.path.split('/file/')
        except ValueError:
            raise InvalidUriError(uri, 'Invalid Girder URI. Girder URIs must '
                                       'contain a /file/<file_id> segment.')

        return cls(file_id, '%s%s' % (scheme, path), api_key, token)

    # NOTE: This usage of api "root" contradicts girder client's notion of the
    #       api root
    def __init__(self, file_id, api_root='http://localhost:8080/api/v1',
                 api_key=None, token=None):
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
        self.token = token
        self.api_key = api_key

        # compat with DataFileElement (compute_many_descriptors needs this)
        self._filepath = self.file_id

        self.gc = girder_client.GirderClient(apiUrl=api_root)

        if token is not None:
            self.gc.token = token
        elif api_key is not None:
            self.gc.authenticate(apiKey=api_key)

        # Cache so we don't have to query server multiple times for multiple
        # calls.
        self._content_type = None

    def __repr__(self):
        return super(GirderDataElement, self).__repr__() + \
            "{file_id: %s, api_root: %s, api_key: %s, token: %s}" % (
                self.file_id, self.api_root, self.api_key or '',
                self.token or ''
            )

    def get_config(self):
        return {
            'file_id': self.file_id,
            'api_root': self.api_root,
            'api_key': self.api_key,
            'token': self.token
        }

    def content_type(self):
        if self._content_type is None:
            self._log.debug("Getting content type for file ID %s"
                            % self.file_id)
            file_model = self.get_file_model()

            if file_model is not None:
                self._content_type = file_model['mimeType']

        return self._content_type

    def get_file_model(self):
        """
        Get the file model json from the server.

        Returns None if the file can't be retrieved from the server.

        :return: file model model as a dictionary
        :rtype: dict | None

        """
        try:
            return self.gc.getFile(self.file_id)
        except girder_client.HttpError:
            return None

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
        """
        # Download file bytes from girder
        content = six.BytesIO()
        self._log.debug("Getting bytes for file ID %s", self.file_id)
        self.gc.downloadFile(self.file_id, content)
        return bytes(content.getvalue())

    def writable(self):
        """
        Determine if a Girder file is able to be written to. Note that this
        requires inferring the access level by traversing to the parent folder
        since this is how Girder determines access.

        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        file_model = self.get_file_model()

        if file_model is None:
            return False
        else:
            item_model = self.gc.getItem(file_model['itemId'])
            folder_model = self.gc.getFolder(item_model['folderId'])

            # See girder.constants.AccessType
            return folder_model['_accessLevel'] >= 1

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
        if not self.writable():
            raise ReadOnlyError('Unauthorized access to write to Girder file %s'
                                % self.file_id)

        try:
            self.gc.uploadFileContents(self.file_id, six.BytesIO(b), len(b))
        except girder_client.HttpError as e:
            if e.status == 401:
                raise ReadOnlyError('Unauthorized access to write to Girder '
                                    'file %s' % self.file_id)
            else:
                raise e
