__author__ = 'purg'

import hashlib
import mimetypes

from smqtk.data_rep import DataElement


class FileElement (DataElement):
    """
    File-based data element
    """

    def __init__(self, filepath):
        """
        Create a new FileElement.

        :param filepath: Path to the file to wrap.
        :type filepath: str

        """
        self._filepath = filepath
        self._content_type = mimetypes.guess_type(filepath)[0]

        # Cache variables for lazy lading
        self._md5_cache = None

    @property
    def uuid(self):
        """
        UUID for this data element. File data elements use the file MD5 sum as
        its UUID.

        :return: UUID value for this data element.
        :rtype: basestring

        """
        if not self._md5_cache:
            self._md5_cache = \
                hashlib.md5(self.get_bytes()).hexdigest()
        return self._md5_cache

    def __hash__(self):
        return self.uuid

    def content_type(self):
        return self._content_type

    def get_bytes(self):
        return open(self._filepath, 'rb').read()


DATA_ELEMENT_CLASS = FileElement
