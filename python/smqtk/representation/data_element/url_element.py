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
        # have to be able to connect to the internet
        try:
            # using github because that's where this repo has been hosted.
            r = requests.get('http://github.com')
            _ = r.content
            return True
        except requests.ConnectionError:
            cls.get_logger().warning(
                "DataUrlElement not usable, cannot connect to "
                "http://github.com"
            )
            return False

    @classmethod
    def from_uri(cls, uri):
        m = cls.URI_RE.match(uri)
        if m is not None:
            # simply pass on URI as URL address
            return DataUrlElement(uri)

        raise InvalidUriError(uri, "Invalid web URI")

    def __init__(self, url_address):
        """
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
