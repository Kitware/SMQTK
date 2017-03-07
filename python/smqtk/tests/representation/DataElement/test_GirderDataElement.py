import mock
import os
import unittest

import nose.tools
import requests

from smqtk.representation.data_element.girder import GirderDataElement
from smqtk.tests import TEST_DATA_DIR
from smqtk.exceptions import ReadOnlyError


DATA_KITWARE_URL = 'https://data.kitware.com'


# Check to see if we have an internet connection.
internet_available = True
try:
    r = requests.get(DATA_KITWARE_URL)
    _ = r.content
except requests.ConnectionError:
    internet_available = False


def gen_response(content, status_code=200):
    resp = requests.Response()
    resp._content = content
    resp.status_code = status_code
    resp.headers['content-length'] = len(content)
    return resp


class TestGirderDataElement (unittest.TestCase):
    """
    Tests for the GirderDataElement plugin implementation

    """

    LOCAL_APIROOT = "http://localhost:8080/api/v1"
    EXAMPLE_ITEM_ID = '5820bbeb8d777f10f26efc2f'
    EXAMPLE_GIRDER_API_ROOT = "%s/api/v1" % DATA_KITWARE_URL
    EXAMPLE_GIRDER_FULL_URI = (
        EXAMPLE_GIRDER_API_ROOT.replace('https', 'girders')
        + '/file/%s' % EXAMPLE_ITEM_ID
    )
    EXAMPLE_GIRDER_SIMPLE_URI = "girder://file:%s" % EXAMPLE_ITEM_ID
    EXAMPLE_PTH = os.path.join(TEST_DATA_DIR, 'Lenna.png')
    EXAMPLE_SHA512 = 'ca2be093f4f25d2168a0afebe013237efce02197bcd8a87f41f' \
                     'f9177824222a1ddae1f6b4f5caf11d68f2959d13f399ea55bd6' \
                     '77c979642d723e02eb9b5dc4d5'

    def test_new_fileId(self):
        expected_id = "some id"
        e = GirderDataElement(expected_id)
        nose.tools.assert_equal(e.file_id, expected_id)
        nose.tools.assert_equal(e.api_root, self.LOCAL_APIROOT)
        nose.tools.assert_is_none(e.token)
        nose.tools.assert_is_none(e.token_expiration)
        nose.tools.assert_is_none(e._content_type)

    def test_repr(self):
        expected_file_id = 'some_file id'
        expected_api_root = 'https://some.server/api/v1'
        expected_api_key = 'someKeyHere'

        e = GirderDataElement(expected_file_id, expected_api_root,
                              expected_api_key)
        actual_repr = repr(e)

        expected_repr = "GirderDataElement{id: some_file id, " \
                        "api_root: https://some.server/api/v1, " \
                        "api_key: someKeyHere}"
        nose.tools.assert_equal(actual_repr, expected_repr)

    def test_configuration_default(self):
        default_config = GirderDataElement.get_default_config()
        nose.tools.assert_equal(default_config,
                                {"file_id": None,
                                 "api_root": self.LOCAL_APIROOT,
                                 "api_key": None})

    def test_from_config_full_constructor(self):
        expected_file_id = '34uhki34gh2345ghjk'
        expected_api_root = 'https://some.other.server/api/v1'
        expected_api_key = '1234ghk135hlg23435'
        new_config = {
            'file_id': expected_file_id,
            'api_root': expected_api_root,
            'api_key': expected_api_key,
        }
        e = GirderDataElement.from_config(new_config)
        nose.tools.assert_equal(e.file_id, expected_file_id)
        nose.tools.assert_equal(e.api_root, expected_api_root)
        nose.tools.assert_equal(e.token_manager.api_key, expected_api_key)
        nose.tools.assert_equal(e.get_config(), new_config)

    def test_from_config_common_partial(self):
        expected_file_id = '5hjkl1345hjk'
        expected_api_root = self.LOCAL_APIROOT
        expected_api_key = None
        e = GirderDataElement.from_config({'file_id': expected_file_id})
        nose.tools.assert_equal(e.file_id, expected_file_id)
        nose.tools.assert_equal(e.api_root, expected_api_root)
        nose.tools.assert_equal(e.token_manager.api_key, expected_api_key)
        nose.tools.assert_equal(e.get_config(),
                                {'file_id': expected_file_id,
                                 'api_root': expected_api_root,
                                 'api_key': expected_api_key})

    def test_from_uri_full_url(self):
        e = GirderDataElement.from_uri(self.EXAMPLE_GIRDER_FULL_URI)
        nose.tools.assert_equal(e.api_root, self.EXAMPLE_GIRDER_API_ROOT)
        nose.tools.assert_equal(e.file_id, self.EXAMPLE_ITEM_ID)
        nose.tools.assert_equal(e.token_manager.api_key, None)

    def test_from_uri_simple_uri(self):
        e = GirderDataElement.from_uri(self.EXAMPLE_GIRDER_SIMPLE_URI)
        nose.tools.assert_equal(e.api_root, self.LOCAL_APIROOT)
        nose.tools.assert_equal(e.file_id, self.EXAMPLE_ITEM_ID)
        nose.tools.assert_equal(e.token_manager.api_key, None)

    def test_from_uri_bad_tag(self):
        # Ensures we catch a bad tag in the URI, i.e., one that is neither
        # girder nor girders.
        nose.tools.assert_raises(ValueError, GirderDataElement.from_uri,
                                 uri='a_bad_tag')

    def test_from_uri_bad_uri(self):
        # Ensures that we catch a bad URI, i.e., one that is of neither form:
        # girder(s)://<user>:<pass>@<host>:<port>/api/v1/file/<file_id>
        # girder://<user>:<pass>@file:<file_id>
        nose.tools.assert_raises(ValueError, GirderDataElement.from_uri,
                                 uri='girder://abcd.com')

    def test_from_uri_bad_path(self):
        # Ensures that we catch a URI that has an appropriate tag and netloc,
        # but the path does not begin with /api, so it is an invalid girder
        # API root.
        nose.tools.assert_raises(ValueError, GirderDataElement.from_uri,
                                 uri='girder://localhost:8080/bad/path')

    def test_from_uri_no_file_in_path(self):
        # Ensures that we catch a URI that has the URL, but doesn't have the
        # /file/<file_id> portion of the URL.
        nose.tools.assert_raises(AssertionError, GirderDataElement.from_uri,
                                 uri='girders://localhost:8080/api/v1/nofile/')

    @mock.patch('smqtk.representation.data_element.girder.requests')
    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    def test_content_type_no_cache(self, m_gtm, m_requests):
        # Mocking such that we simulate a valid API root and an existing
        # item reference

        # Dummy requests return value
        expected_mimetype = 'some/type'
        m_requests.get('setting json mock return value')\
            .json.return_value = {
                'mimeType': expected_mimetype
            }

        e = GirderDataElement('foo')
        actual_type = e.content_type()
        nose.tools.assert_equal(actual_type, expected_mimetype)
        # once above to set return_value and once it function
        nose.tools.assert_equal(m_requests.get.call_count, 2)
        m_requests.get().json.assert_called_once()

    @mock.patch('smqtk.representation.data_element.girder.requests')
    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    def test_content_type_cached(self, m_gtm, m_requests):
        expected_mimetype = 'some/type'

        e = GirderDataElement('id')
        e._content_type = expected_mimetype

        actual_type = e.content_type()
        nose.tools.assert_equal(actual_type, expected_mimetype)
        nose.tools.assert_equal(m_requests.get.call_count, 0)
        nose.tools.assert_equal(m_requests.get().json.call_count, 0)

    @mock.patch('smqtk.representation.data_element.girder.requests')
    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    def test_get_file_model(self, m_gtm, m_requests):
        # Check static expected values in model. This happens to be actual
        # return values, however we are mocking out any actual network
        # communication.
        expected_m = {
            '_id': self.EXAMPLE_ITEM_ID,
            '_modelType': 'file',
            'exts': ['png'],
            'mimeType': 'image/png',
            'name': 'Lenna.png',
            'sha512': self.EXAMPLE_SHA512,
            'size': 473831
        }

        # as if a successful call
        m_requests.get().configure_mock(status_code=200)
        m_requests.get().json.return_value = expected_m

        e = GirderDataElement(self.EXAMPLE_ITEM_ID,
                              self.EXAMPLE_GIRDER_API_ROOT)
        m = e.get_file_model()

        nose.tools.assert_equal(m['_id'], expected_m['_id'])
        nose.tools.assert_equal(m['_modelType'], expected_m['_modelType'])
        nose.tools.assert_equal(m['exts'], expected_m['exts'])
        nose.tools.assert_equal(m['mimeType'], expected_m['mimeType'])
        nose.tools.assert_equal(m['name'], expected_m['name'])
        nose.tools.assert_equal(m['sha512'], expected_m['sha512'])
        nose.tools.assert_equal(m['size'], expected_m['size'])

    @mock.patch('smqtk.representation.data_element.girder.requests')
    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    def test_get_file_model_item_no_exists(self, m_gtm, m_requests):
        # i.e. status 400 returned, should return None
        m_requests.get().configure_mock(status_code=400)
        e = GirderDataElement('id')
        m = e.get_file_model()
        nose.tools.assert_is_none(m)
        nose.tools.assert_equal(m_requests.get().raise_for_status.call_count,
                                0)

    @mock.patch('smqtk.representation.data_element.girder.requests')
    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    def test_get_file_model_other_bad_status(self, m_gtm, m_requests):
        # if some other bad status code is returned an exception is raised

        def r(*args, **kwds):
            raise requests.HTTPError('message')

        m_requests.get().configure_mock(status_code=502)
        m_requests.get().raise_for_status.side_effect = r

        e = GirderDataElement('id')
        nose.tools.assert_raises(
            requests.HTTPError,
            e.get_file_model
        )
        m_requests.get().raise_for_status.assert_called_once()

    def test_is_empty_none_model(self):
        # Uses model return, empty if no model return (no item in girder by ID)
        e = GirderDataElement('someId')
        e.get_file_model = mock.MagicMock(return_value=None)
        nose.tools.assert_true(e.is_empty())

    def test_is_empty_zero_size(self):
        # Uses model return size parameter
        e = GirderDataElement('someId')
        e.get_file_model = mock.MagicMock(return_value={'size': 0})
        nose.tools.assert_true(e.is_empty())

    def test_is_empty_nonzero_bytes(self):
        e = GirderDataElement('someId')
        e.get_file_model = mock.MagicMock(return_value={'size': 7})
        nose.tools.assert_false(e.is_empty())

    def test_writable(self):
        # Currently a read-only element
        nose.tools.assert_false(
            GirderDataElement('someId').writable()
        )

    def test_set_bytes(self):
        # Currently a read-only element
        e = GirderDataElement('someId')
        nose.tools.assert_raises(
            ReadOnlyError,
            e.set_bytes,
            'bytes'
        )

    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    @mock.patch('smqtk.representation.data_element.girder.requests.get')
    def test_get_bytes(self, m_requests_get, m_gtm):
        expected_content = 'Some byte content.'
        expected_response = gen_response(expected_content)

        # Mock out requests call
        m_requests_get.return_value = expected_response

        # Get mocked bytes
        e = GirderDataElement('someId')
        actual_bytes = e.get_bytes()

        nose.tools.assert_equal(expected_content, actual_bytes)

    @mock.patch('smqtk.representation.data_element.girder.GirderTokenManager')
    @mock.patch('smqtk.representation.data_element.girder.requests.get')
    def test_get_bytes_bad_response(self, m_requests_get, m_gtm):
        expected_content = 'Some byte content.'
        expected_statCode = 501
        expected_response = gen_response(expected_content,
                                         expected_statCode)

        # Mock out requests call
        m_requests_get.return_value = expected_response

        # Try getting mocked bytes
        e = GirderDataElement('someId')
        nose.tools.assert_raises_regexp(
            requests.HTTPError,
            "%d" % expected_statCode,
            e.get_bytes
        )
