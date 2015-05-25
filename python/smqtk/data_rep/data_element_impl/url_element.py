__author__ = 'purg'

import hashlib
import mimetypes
import requests

from smqtk.data_rep import DataElement


MIMETYPES = mimetypes.MimeTypes()


class DataUrlElement (DataElement):
    """
    Representation of data loadable via a web URL address.
    """

    def __init__(self, url_address):
        self._url = url_address
        self._md5_cache = None

        # make sure that url has a http:// or https:// prefix
        if not (self._url[:7] == "http://" or self._url[:8] == "https://"):
            self._url = "http://" + self._url

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        # return MIMETYPES.guess_type(self._url)[0]
        return requests.get(self._url).headers['content-type']

    def md5(self):
        """
        :return: MD5 hex string of the data content.
        :rtype: str
        """
        if not self._md5_cache:
            self._md5_cache = hashlib.md5(self.get_bytes()).hexdigest()
        return self._md5_cache

    def uuid(self):
        """
        UUID for this data element.

        URL elements use the byte's MD5 sum as the UUID.

        :return: UUID value for this data element. This return value should be
            hashable.
        :rtype: collections.Hashable

        """
        return self.md5()

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes

        :raises requests.exceptions.HTTPError: Error during request for data
            via GET.

        """
        # Fetch content from URL, return bytes
        r = requests.get(self._url)
        r.raise_for_status()
        if r.ok:
            return r.content
        else:
            raise RuntimeError("Request response not OK. Status code returned: "
                               "%d", r.status_code)
