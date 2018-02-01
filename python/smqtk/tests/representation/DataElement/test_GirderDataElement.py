import mock
import os
import unittest

import nose.tools
import requests

from smqtk.exceptions import InvalidUriError
from smqtk.representation.data_element.girder import (
    GirderDataElement,
    girder_client  # None when not available
)
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


# Only perform these tests if the implementation is usable.
if GirderDataElement.is_usable():

    class TestGirderDataElement (unittest.TestCase):
        """
        Tests for the GirderDataElement plugin implementation

        """

        LOCAL_APIROOT = "http://localhost:8080/api/v1"
        EXAMPLE_ITEM_ID = '5820bbeb8d777f10f26efc2f'
        EXAMPLE_GIRDER_API_ROOT = "%s/api/v1" % DATA_KITWARE_URL
        EXAMPLE_GIRDER_FULL_URI = (
            EXAMPLE_GIRDER_API_ROOT.replace('https', 'girder')
            + '/file/%s' % EXAMPLE_ITEM_ID
        )
        EXAMPLE_PTH = os.path.join(TEST_DATA_DIR, 'Lenna.png')
        EXAMPLE_SHA512 = 'ca2be093f4f25d2168a0afebe013237efce02197bcd8a87f41f' \
                         'f9177824222a1ddae1f6b4f5caf11d68f2959d13f399ea55bd6' \
                         '77c979642d723e02eb9b5dc4d5'

        def test_new_fileId(self):
            expected_id = "some id"
            e = GirderDataElement(expected_id)
            nose.tools.assert_equal(e.file_id, expected_id)
            nose.tools.assert_equal(e.api_root, self.LOCAL_APIROOT)
            nose.tools.assert_is_none(e.api_key)
            nose.tools.assert_is_none(e.token)
            nose.tools.assert_is_none(e._content_type)
            nose.tools.assert_is_instance(e.gc, girder_client.GirderClient)

        @mock.patch('girder_client.GirderClient.authenticate')
        def test_repr(self, mock_requests):
            expected_file_id = 'some_file id'
            expected_api_root = 'https://some.server/api/v1'
            expected_api_key = 'someKeyHere'

            e = GirderDataElement(expected_file_id, expected_api_root,
                                  expected_api_key)
            actual_repr = repr(e)

            expected_repr = "GirderDataElement{file_id: %s, " \
                            "api_root: %s, " \
                            "api_key: %s, token: }" % (
                                expected_file_id, expected_api_root, expected_api_key)
            nose.tools.assert_equal(actual_repr, expected_repr)

        def test_configuration_default(self):
            default_config = GirderDataElement.get_default_config()
            nose.tools.assert_equal(default_config,
                                    {"file_id": None,
                                     "api_root": self.LOCAL_APIROOT,
                                     "api_key": None,
                                     "token": None})

        @mock.patch('girder_client.GirderClient.authenticate')
        def test_from_config_full_constructor(self, mock_authenticate):
            expected_file_id = '34uhki34gh2345ghjk'
            expected_api_root = 'https://some.other.server/api/v1'
            expected_api_key = '1234ghk135hlg23435'
            new_config = {
                'token': None,
                'file_id': expected_file_id,
                'api_root': expected_api_root,
                'api_key': expected_api_key,
            }
            e = GirderDataElement.from_config(new_config)
            nose.tools.assert_equal(e.file_id, expected_file_id)
            nose.tools.assert_equal(e.api_root, expected_api_root)
            nose.tools.assert_equal(e.get_config(), new_config)

        def test_from_config_common_partial(self):
            expected_file_id = '5hjkl1345hjk'
            expected_api_root = self.LOCAL_APIROOT
            expected_api_key = None
            e = GirderDataElement.from_config({'file_id': expected_file_id})
            nose.tools.assert_equal(e.file_id, expected_file_id)
            nose.tools.assert_equal(e.api_root, expected_api_root)
            nose.tools.assert_equal(e.get_config(),
                                    {'file_id': expected_file_id,
                                     'api_root': expected_api_root,
                                     'api_key': expected_api_key,
                                     'token': None})

        def test_from_uri_full_url(self):
            e = GirderDataElement.from_uri(self.EXAMPLE_GIRDER_FULL_URI)
            nose.tools.assert_equal(e.file_id, self.EXAMPLE_ITEM_ID)

        def test_from_uri_bad_tag(self):
            # Ensures we catch a bad tag in the URI, i.e., one that is neither
            # girder nor girders.
            nose.tools.assert_raises(InvalidUriError, GirderDataElement.from_uri,
                                     uri='a_bad_tag')

        def test_from_uri_bad_path(self):
            # Ensures that we catch a URI that has an appropriate tag and netloc,
            # but the path does not begin with /api, so it is an invalid girder
            # API root.
            nose.tools.assert_raises(InvalidUriError, GirderDataElement.from_uri,
                                     uri='girder://localhost:8080/bad/path')

        @mock.patch('girder_client.GirderClient.getFile')
        def test_content_type_no_cache(self, m_getFile):
            # Mocking such that we simulate a valid API root and an existing
            # item reference

            # Dummy requests return value
            expected_mimetype = 'some/type'
            m_getFile.return_value = {
                    'mimeType': expected_mimetype
            }

            e = GirderDataElement('foo')
            actual_type = e.content_type()
            nose.tools.assert_equal(actual_type, expected_mimetype)
            m_getFile.assert_called_once()

            # Ensure that calling content_type a second time doesn't call getFile again
            nose.tools.assert_equal(e.content_type(), expected_mimetype)
            m_getFile.assert_called_once()

        @mock.patch('girder_client.GirderClient.getFile')
        def test_get_file_model(self, m_getFile):
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

            m_getFile.return_value = expected_m

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

        @mock.patch('girder_client.GirderClient.getFile')
        def test_get_file_model_item_no_exists(self, m_getFile):
            def raise_http_error(*args, **kwargs):
                raise girder_client.HttpError(None, None, None, None)
            m_getFile.side_effect = raise_http_error

            e = GirderDataElement('foo', self.EXAMPLE_GIRDER_API_ROOT)
            m = e.get_file_model()
            nose.tools.assert_is_none(m)

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

        @mock.patch('smqtk.representation.data_element.girder.GirderDataElement.get_file_model')
        @mock.patch('girder_client.GirderClient.getFolder')
        @mock.patch('girder_client.GirderClient.getItem')
        def test_writable(self, m_getItem, m_getFolder, m_get_file_model):
            m_getItem.return_value = {'folderId': 'someFolderId'}
            m_getFolder.return_value = {'_accessLevel': 1}

            nose.tools.assert_true(GirderDataElement('someId').writable())

            # Access level 0 should cause it to be unwritable
            m_getFolder.return_value = { '_accessLevel': 0 }
            nose.tools.assert_false(GirderDataElement('someId').writable())

            # A nonexistent file model should make writable return false
            gde = GirderDataElement('someId')
            gde.get_file_model = mock.MagicMock(return_value=None)
            nose.tools.assert_false(gde.writable())

        @mock.patch('girder_client.GirderClient.uploadFileContents')
        def test_set_bytes(self, m_uploadFileContents):
            gde = GirderDataElement('someId')
            gde.writable = mock.MagicMock(return_value=True)
            gde.set_bytes(b'foo')
            m_uploadFileContents.assert_called_once()

        def test_set_bytes_non_writable(self):
            gde = GirderDataElement('someId')
            gde.writable = mock.MagicMock(return_value=False)
            nose.tools.assert_raises(ReadOnlyError, gde.set_bytes, b=None)

        def test_set_bytes_http_errors(self):
            gde = GirderDataElement('someId')
            gde.writable = mock.MagicMock(return_value=True)

            # Test access denied throws ReadOnlyError
            gde.gc.uploadFileContents = mock.MagicMock(side_effect=girder_client.HttpError(401, '', None, None))
            nose.tools.assert_raises(ReadOnlyError, gde.set_bytes, b=b'foo')

            # Test any other error (like a 500) re-raises the HttpError
            gde.gc.uploadFileContents = mock.MagicMock(side_effect=girder_client.HttpError(500, '', None, None))
            nose.tools.assert_raises(girder_client.HttpError, gde.set_bytes, b=b'foo')

        @mock.patch('girder_client.GirderClient.downloadFile')
        @mock.patch('six.BytesIO.getvalue')
        def test_get_bytes(self, m_getvalue, m_downloadFile):
            m_getvalue.return_value = 'foo'

            e = GirderDataElement('someId')
            actual_bytes = e.get_bytes()

            nose.tools.assert_equal(actual_bytes, bytes('foo'))
