__author__ = 'purg'

import base64
import hashlib

from smqtk.data_rep import DataElement


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
        self._bytes = bytes
        self._content_type = content_type
        self._md5 = hashlib.md5(self._bytes).hexdigest()

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return self._content_type

    def md5(self):
        """
        :return: MD5 hex string of the data content.
        :rtype: str
        """
        return self._md5

    def uuid(self):
        """
        UUID for this data element.

        Memory elements use the byte's MD5 sum as the UUID.

        :return: UUID value for this data element. This return value should be
            hashable.
        :rtype: collections.Hashable

        """
        return self._md5

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return self._bytes
