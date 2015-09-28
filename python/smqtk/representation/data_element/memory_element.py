
import base64
import hashlib

from smqtk.representation import DataElement


__author__ = "paul.tunison@kitware.com"


class DataMemoryElement (DataElement):
    """
    In-memory representation of data stored in a byte list
    """

    @classmethod
    def from_base64(cls, b64_str, content_type):
        """
        Create new MemoryElement instance based on a given base64 string and
        content type.

        :param b64_str: Base64 data string.
        :type b64_str: str

        :param content_type: Content type string
        :type content_type: str

        :return: New MemoryElement instance containing the byte data in the
            given base64 string.
        :rtype: DataMemoryElement

        """
        return DataMemoryElement(base64.decodestring(b64_str), content_type)

    def __init__(self, bytes, content_type):
        super(DataMemoryElement, self).__init__()

        self._bytes = bytes
        self._content_type = content_type

        # since we have the bytes right now, short circuiting the checksum
        # caches
        self._md5_cache = hashlib.md5(self._bytes).hexdigest()
        self._sha1_cache = hashlib.sha1(self._bytes).hexdigest()

    def get_config(self):
        return {
            "bytes": self._bytes,
            'content_type': self._content_type,
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return self._content_type

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return self._bytes


DATA_ELEMENT_CLASS = DataMemoryElement
