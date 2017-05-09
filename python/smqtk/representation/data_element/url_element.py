import mimetypes
import re
import requests

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation import DataElement


MIMETYPES = mimetypes.MimeTypes()


class DataUrlElement (DataElement):
    """
    Representation of data loadable via a web URL address.
    """

    # Enforce presence of demarcating schema
    URI_RE = re.compile('^https?://.+$')

    @classmethod
    def is_usable(cls):
        # URLs are not necessarily on the public internet. Local networking
        # should always be available.
        return True

    @classmethod
    def from_uri(cls, uri):
        m = cls.URI_RE.match(uri)
        if m is not None:
            # simply pass on URI as URL address
            return DataUrlElement(uri)

        raise InvalidUriError(uri, "Invalid web URI")

    def __init__(self, url_address):
        """
        Create a new URL element for a URL address.

        The given address may not resolve to anything.

        :raises requests.exceptions.ConnectionError: Failed to connect with the
            given hostname.
        :raises requests.exceptions.HTTPError: URL address provided does not
            resolve into a valid GET request.

        :param url_address: Web address of element
        :type url_address: str

        """
        super(DataUrlElement, self).__init__()

        self._url = url_address

        # make sure that url has a http:// or https:// prefix
        if not (self._url[:7] == "http://" or self._url[:8] == "https://"):
            self._url = "http://" + self._url

        # Check that the URL is valid, i.e. actually points to something
        requests.get(self._url).raise_for_status()

    def __repr__(self):
        return super(DataUrlElement, self).__repr__() + "{url: %s}" % self._url

    def get_config(self):
        return {
            "url_address": self._url
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return requests.get(self._url).headers['content-type']

    def is_empty(self):
        """
        Check if this element contains no bytes.

        :return: If this element contains 0 bytes.
        :rtype: bool

        """
        return len(self.get_bytes()) == 0

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
        return r.content

    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        # Web addresses cannot be written to
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
        raise ReadOnlyError("URL address targets cannot be written to.")


DATA_ELEMENT_CLASS = DataUrlElement
