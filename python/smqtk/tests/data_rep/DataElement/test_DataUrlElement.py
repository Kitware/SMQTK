
import nose.tools as ntools
import unittest

from smqtk.data_rep.data_element_impl.url_element import DataUrlElement


__author__ = 'purg'


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
