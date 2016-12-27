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


if internet_available:

    # Public domain Lenna image from Wikipedia, same as local test image file
    EXAMPLE_URL = 'https://data.kitware.com/api/v1/file/5820bbeb8d777f10f26efc2f/download'
    EXAMPLE_PTH = os.path.join(TEST_DATA_DIR, 'Lenna.png')


    class TestDataUrlElement (unittest.TestCase):
        """
        :NOTE: These tests require a connection to the internet in order to
        pass.
        """

        @mock.patch('requests.get')
        def test_is_usable(self, m_req_get):
            # mocked function returns a Mock object that has mocked attribute
            # ``content``.
            ntools.assert_true(DataUrlElement.is_usable())

        @mock.patch('requests.get')
        def test_is_not_usable(self, m_req_get):
            # Pretend the get function returns a Connection error, which would
            # happen if URL resolution wasn't occurring.
            def r(*args, **kwds):
                raise requests.ConnectionError()
            m_req_get.side_effect = r
            ntools.assert_false(DataUrlElement.is_usable())

        def test_new(self):
            e = DataUrlElement(EXAMPLE_URL)

        def test_new_add_missing_scheme(self):
            # Construct without scheme header, should add http://
            e = DataUrlElement(EXAMPLE_URL[8:])
            ntools.assert_equal(e._url, 'http://' + EXAMPLE_URL[8:])

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

            inst2 = DataUrlElement.from_config(inst1.get_config())
            ntools.assert_equal(inst1._url, inst2._url)

        def test_from_uri(self):
            e = DataUrlElement.from_uri(EXAMPLE_URL)
            ntools.assert_equal(e._url, EXAMPLE_URL)

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

        def test_get_bytes(self):
            e = DataUrlElement(EXAMPLE_URL)
            ntools.assert_equal(e.get_bytes(), open(EXAMPLE_PTH).read())
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

        @mock.patch('smqtk.representation.data_element.url_element.requests.get')
        def test_writable_readonly(self, m_requests_get):
            e = DataUrlElement('')
            ntools.assert_false(e.writable())

        @mock.patch('smqtk.representation.data_element.url_element.requests.get')
        def test_set_bytes_readonly(self, m_requests_get):
            e = DataUrlElement('')
            ntools.assert_raises(
                ReadOnlyError,
                e.set_bytes, 'some-bytes'
            )
