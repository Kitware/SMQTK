import nose.tools as ntools
import os
import requests
import unittest

from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.tests import TEST_DATA_DIR


# Check to see if we have an internet connection.
internet_available = True
try:
    r = requests.get('https://data.kitware.com')
    _ = r.content
except requests.ConnectionError:
    internet_available = False


if internet_available:

    # Public domain Lenna image from Wikipedia, same as local test image file
    EXAMPLE_URL = 'https://data.kitware.com/api/v1/file/5820bbeb8d777f10f26efc2f/download'
    EXAMPLE_PTH = os.path.join(TEST_DATA_DIR, 'Lenna.png')


    class TestDataUrlElement (unittest.TestCase):
        """

        :NOTE: These tests require a connection to the internet in order to pass.

        """

        def test_new(self):
            e = DataUrlElement(EXAMPLE_URL)
            ntools.assert_equal(e.get_bytes(), open(EXAMPLE_PTH).read())

        def test_content_type(self):
            e = DataUrlElement(EXAMPLE_URL)
            ntools.assert_equal(e.content_type(), 'image/png')

        def test_from_configuration(self):
            default_config = DataUrlElement.get_default_config()
            ntools.assert_equal(default_config, {'url_address': None})

            default_config['url_address'] = EXAMPLE_URL
            inst1 = DataUrlElement.from_config(default_config)
            ntools.assert_equal(default_config, inst1.get_config())
            ntools.assert_equal(inst1._url, EXAMPLE_URL)
            ntools.assert_equal(inst1.get_bytes(), open(EXAMPLE_PTH).read())

            inst2 = DataUrlElement.from_config(inst1.get_config())
            ntools.assert_equal(inst1, inst2)
