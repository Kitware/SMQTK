import mock
import nose.tools as ntools
import os
import requests
import unittest

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.tests import TEST_DATA_DIR


# Check to see if we have an internet connection.
internet_available = True
try:
    r = requests.get('https://data.kitware.com')
    _ = r.content
except requests.ConnectionError:
    internet_available = False


class TestDataUrlElement (unittest.TestCase):
    """

    :NOTE: Some of these tests require a connection to the internet in order to
        pass.

    """

    # Public domain Lenna image from Wikipedia, same as local test image file
    EXAMPLE_URL = \
        'https://data.kitware.com/api/v1/file/5820bbeb8d777f10f26efc2f/download'
    EXAMPLE_PTH = os.path.join(TEST_DATA_DIR, 'Lenna.png')

    def test_is_usable(self):
        # Should always be available, because local/intranet networks are a
        # thing.
        ntools.assert_true(DataUrlElement.is_usable())

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_new_from_internet(self):
        e = DataUrlElement(self.EXAMPLE_URL)
        ntools.assert_equal(e.get_bytes(), open(self.EXAMPLE_PTH, 'rb').read())

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_new_add_missing_scheme(self):
        # Construct without scheme header, should add http://
        e = DataUrlElement(self.EXAMPLE_URL[8:])
        ntools.assert_equal(e._url, 'http://' + self.EXAMPLE_URL[8:])
        ntools.assert_equal(e.get_bytes(), open(self.EXAMPLE_PTH, 'rb').read())

    def test_new_invalid_url(self):
        ntools.assert_raises(
            requests.ConnectionError,
            DataUrlElement,
            'http://not.a.real.host/'
        )

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_content_type(self):
        e = DataUrlElement(self.EXAMPLE_URL)
        ntools.assert_equal(e.content_type(), 'image/png')

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_from_configuration(self):
        default_config = DataUrlElement.get_default_config()
        ntools.assert_equal(default_config, {'url_address': None})

        default_config['url_address'] = self.EXAMPLE_URL
        inst1 = DataUrlElement.from_config(default_config)
        ntools.assert_equal(default_config, inst1.get_config())
        ntools.assert_equal(inst1._url, self.EXAMPLE_URL)
        ntools.assert_equal(inst1.get_bytes(),
                            open(self.EXAMPLE_PTH, 'rb').read())

        inst2 = DataUrlElement.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_from_uri(self):
        e = DataUrlElement.from_uri(self.EXAMPLE_URL)
        ntools.assert_equal(e.get_bytes(), open(self.EXAMPLE_PTH, 'rb').read())
        ntools.assert_equal(e.content_type(), 'image/png')

    def test_from_uri_no_scheme(self):
        ntools.assert_raises(
            InvalidUriError,
            DataUrlElement.from_uri,
            'www.kitware.com'
        )

    def test_from_uri_invalid_scheme(self):
        ntools.assert_raises(
            InvalidUriError,
            DataUrlElement.from_uri,
            'ftp://www.kitware.com'
        )

    @mock.patch('smqtk.representation.data_element.url_element.requests.get')
    def test_is_empty_zero_bytes(self, m_requests_get):
        e = DataUrlElement('some-address')
        # simulate no content bytes returned
        e.get_bytes = mock.MagicMock(return_value='')
        ntools.assert_true(e.is_empty())

    @mock.patch('smqtk.representation.data_element.url_element.requests.get')
    def test_is_empty_nonzero_bytes(self, m_requests_get):
        e = DataUrlElement('some-address')
        # simulate some content bytes returned
        e.get_bytes = mock.MagicMock(return_value='some bytes returned')
        ntools.assert_false(e.is_empty())

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_get_bytes_from_url(self):
        e = DataUrlElement(self.EXAMPLE_URL)
        ntools.assert_equal(e.get_bytes(), open(self.EXAMPLE_PTH, 'rb').read())
        ntools.assert_equal(e.content_type(), 'image/png')

    @mock.patch('smqtk.representation.data_element.url_element.requests.get')
    def test_get_bytes_404_return_code(self, m_requests_get):
        e = DataUrlElement('some-address')

        sim_rc = 500
        simulated_r = requests.Response()
        simulated_r.status_code = sim_rc
        m_requests_get.return_value = simulated_r

        ntools.assert_raises_regexp(
            requests.HTTPError,
            '%d' % sim_rc,
            e.get_bytes
        )

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_writable_universal_false(self):
        # Cannot write to a URL hosted file, should always return false
        e = DataUrlElement(self.EXAMPLE_URL)
        ntools.assert_false(e.writable())

    @unittest.skipUnless(internet_available, "Internet not accessible")
    def test_set_bytes_universal_readonly(self):
        # should never be able to write to a url element
        e = DataUrlElement(self.EXAMPLE_URL)
        ntools.assert_raises(
            ReadOnlyError,
            e.set_bytes, 'foo'
        )
