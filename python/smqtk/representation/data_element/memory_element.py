import base64
import re

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation import DataElement
from smqtk.utils.dict import merge_dict


BYTES_CONFIG_ENCODING = 'latin-1'


class DataMemoryElement (DataElement):
    """
    In-memory representation of data stored in a byte list
    """

    # Base64 RE including URL-safe character replacements
    B64_PATTERN = '[a-zA-Z0-9+/_-]*={0,2}'
    URI_B64_RE = re.compile('^base64://(?P<base64>{})$'.format(B64_PATTERN))
    URI_DATA_B64_RE = re.compile(r"^data:(?P<ct>[\w/]+);base64,(?P<base64>{})$"
                                 .format(B64_PATTERN))

    @classmethod
    def is_usable(cls):
        # No dependencies
        return True

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        Overrides base because this implementation's "bytes" argument wants to
        be given a ``bytes`` type object. When not None, in python 2 this is a
        normal string (not unicode), while in python 3 bytes is a distinct
        type.
        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(),
                                     config_dict)
        try:
            # In python 3, encode input ``str`` into ``bytes``.
            # In python 2, even though ``str`` and ``bytes`` are the same
            # underlying type, we could be given ``unicode``, which needs to be
            # encoded down to ``bytes`` (``str``).
            config_dict["bytes"] = \
                config_dict['bytes'].encode(BYTES_CONFIG_ENCODING)
        except AttributeError:
            # If this is a None value, which has no attributes at all, leave it
            # alone. If in python 2 and given a unicode string, as is the norm
            # return from ``json.load`` and ``json.loads``,
            pass
        return super(DataMemoryElement, cls).from_config(config_dict,
                                                         merge_default=False)

    @classmethod
    def from_uri(cls, uri):
        """
        Construct a new instance based on the given URI.

        Memory elements resolve byte-string formats. Currently, this method
        accepts a base64 using the standard and URL-safe alphabet as the python
        ``base64.urlsafe_b64decode`` module function would expect.

        This method accepts URIs in two formats:
            - ``base64://<data>``
            - ``data:<mimetype>;base64,<data>``
            - Empty string (no data)

        Filling in ``<data>`` with the actual byte string, and ``<mimetype>``
        with the actual MIMETYPE of the bytes.

        :param uri: URI string to resolve into an element instance
        :type uri: str

        :raises smqtk.exceptions.InvalidUriError: The given URI was not a
            base64 format

        :return: New element instance of our type.
        :rtype: DataElement

        """
        if uri is None:
            raise InvalidUriError(uri, 'None value given')

        if len(uri) == 0:
            return DataMemoryElement(b'', None)

        data_b64_m = cls.URI_B64_RE.match(uri)
        if data_b64_m is not None:
            m_d = data_b64_m.groupdict()
            return DataMemoryElement.from_base64(m_d['base64'], None)

        data_b64_m = cls.URI_DATA_B64_RE.match(uri)
        if data_b64_m is not None:
            m_d = data_b64_m.groupdict()
            return DataMemoryElement.from_base64(
                m_d['base64'], m_d['ct']
            )

        raise InvalidUriError(uri, "Did not detect byte format URI")

    @classmethod
    def from_base64(cls, b64_str, content_type=None):
        """
        Create new MemoryElement instance based on a given base64 string and
        content type.

        This method accepts a base64 using the standard and URL-safe alphabet
        as the python ``base64.urlsafe_b64decode`` module function would
        expect.

        :param b64_str: Base64 data string.
        :type b64_str: str

        :param content_type: Content type string, or None if unknown.
        :type content_type: str | None

        :return: New MemoryElement instance containing the byte data in the
            given base64 string.
        :rtype: DataMemoryElement

        """
        if b64_str is None:
            raise ValueError("Base 64 string should not be None")
        # The decode function does not like taking unicode strings
        # (python 2.x). Additionally, the encoding alphabet should not
        # container any unicode symbols, so this aught to be safe.
        b64_str = str(b64_str)
        return DataMemoryElement(base64.urlsafe_b64decode(b64_str),
                                 content_type)

    @staticmethod
    def _assert_is_bytes(v):
        """
        Assert that the value passed in is a bytes-line object or None.

        We co-opt the use of the memoryview constructor to perform the checking
        for us.

        :param v: Some value to check.
        :raises TypeError: The input value ``v`` was not of a bytes-line type.
        """
        if v is not None:
            # Check that the input is "bytes-like" by attempting construction
            # of a memoryview, which is fast and requires a bytes-like input.
            # We don't want to retain it, just boot-strap its error checking.
            memoryview(v)

    # noinspection PyShadowingBuiltins
    def __init__(self, bytes=None, content_type=None, readonly=False):
        """
        Create a new DataMemoryElement from a byte string and optional content
        type.

        :param bytes: Bytes to contain. May be None to represent no bytes.
        :type bytes: None | bytes

        :param content_type: Content type of the bytes given.
        :type content_type: None | basestring

        :param readonly: If this element should allow writing or not.
        :type readonly: bool

        """
        super(DataMemoryElement, self).__init__()
        self._assert_is_bytes(bytes)
        self._bytes = bytes
        self._content_type = content_type
        self._readonly = bool(readonly)

    def __repr__(self):
        return super(DataMemoryElement, self).__repr__() + \
               "{len(bytes): %d, content_type: %s, readonly: %s}" \
               % (len(self.get_bytes()), self._content_type, self._readonly)

    def get_config(self):
        b = self._bytes
        try:
            b = b.decode(BYTES_CONFIG_ENCODING)
        except AttributeError:
            # if ``b`` is None.
            pass
        return {
            "bytes": b,
            'content_type': self._content_type,
            "readonly": self._readonly,
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return self._content_type

    def is_empty(self):
        """
        Check if this element contains no bytes.

        :return: If this element contains 0 bytes.
        :rtype: bool

        """
        return not bool(self._bytes)

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return self._bytes or b''

    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        return not self._readonly

    def set_bytes(self, b):
        """
        Set bytes to this data element in the form of a string.

        Previous content type value is maintained.

        :param b: bytes to set.
        :type b: bytes

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.

        """
        if not self._readonly:
            self._assert_is_bytes(b)
            self._bytes = b
        else:
            raise ReadOnlyError("This memory element cannot be written to.")


DATA_ELEMENT_CLASS = DataMemoryElement
