import logging
import mimetypes
import requests

from smqtk.representation import DataElement


__author__ = "paul.tunison@kitware.com"


MIMETYPES = mimetypes.MimeTypes()


class DataUrlElement (DataElement):
    """
    Representation of data loadable via a web URL address.
    """

    @classmethod
    def is_usable(cls):
        # have to be able to connect to the internet
        try:
            r = requests.get('http://github.com')
            _ = r.content
            return True
        except Exception, ex:
            logging.getLogger("DataUrlElement").warning(
                "DataUrlElement not usable, cannot connect to github.com"
            )
            return False

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


DATA_ELEMENT_CLASS = DataUrlElement
