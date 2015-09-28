
import nose.tools as ntools
import requests
import socket
import unittest

from smqtk.representation.data_element.url_element import DataUrlElement


__author__ = "paul.tunison@kitware.com"


# Check to see if we have an internet connection (by checking github.com
# website resolution).
internet_available = True
try:
    r = requests.get('http://github.com')
    _ = r.content
except Exception, ex:
    internet_available = False


if internet_available:

    # Public domain rose image
    EXAMPLE_URL = \
        'http://www.public-domain-photos.com/free-stock-photos-4/flowers/yellow-rose-3.jpg'


    class TestDataUrlElement (unittest.TestCase):
        """

        :NOTE: These tests require a connection to the internet in order to pass.

        """

        def test_configuration(self):
            default_config = DataUrlElement.get_default_config()
            ntools.assert_equal(default_config, {'url_address': None})

            default_config['url_address'] = EXAMPLE_URL
            inst1 = DataUrlElement.from_config(default_config)
            ntools.assert_equal(default_config, inst1.get_config())
            ntools.assert_equal(inst1._url, EXAMPLE_URL)

            inst2 = DataUrlElement.from_config(inst1.get_config())
            ntools.assert_equal(inst1, inst2)
