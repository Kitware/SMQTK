import base64
import json
import unittest.mock as mock
import os
import unittest

from smqtk.iqr import IqrSession
from smqtk.representation import DescriptorElement
from smqtk.representation.classification_element.memory \
    import MemoryClassificationElement
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement
from smqtk.utils.plugin import Pluggable
from smqtk.web.iqr_service import IqrService

from tests.web.iqr_service.stubs import \
    STUB_MODULE_PATH, \
    StubClassifier, StubDescriptorSet, StubDescrGenerator, \
    StubNearestNeighborIndex


class TestIqrService (unittest.TestCase):

    # Patch in this module for stub implementation access.
    # noinspection PyUnresolvedReferences
    @mock.patch.dict(os.environ, {
        Pluggable.PLUGIN_ENV_VAR: STUB_MODULE_PATH
    })
    def setUp(self):
        """
        Make an instance of the IqrService flask application with stub
        algorithms.
        """
        # Setup configuration for test application
        config = IqrService.get_default_config()
        plugin_config = config['iqr_service']['plugins']

        # Use basic in-memory representation types.
        plugin_config['classification_factory']['type'] =\
            'MemoryClassificationElement'
        plugin_config['descriptor_factory']['type'] = \
            'DescriptorMemoryElement'
        plugin_config['descriptor_set']['type'] = 'StubDescriptorSet'

        # Set up dummy algorithm types
        plugin_config['classifier_config']['type'] = 'StubClassifier'
        plugin_config['descriptor_generator']['type'] = 'StubDescrGenerator'
        plugin_config['neighbor_index']['type'] = 'StubNearestNeighborIndex'

        self.app = IqrService(config)

    def assertStatusCode(self, r, code):
        """
        :type r: :type: flask.wrappers.Response
        :type code: int
        """
        self.assertEqual(code, r.status_code)

    def assertJsonMessageRegex(self, r, regex):
        """
        :type r: flask.wrappers.Response
        :type regex: str
        """
        self.assertRegex(json.loads(r.data.decode())['message'], regex)

    # Test Methods ############################################################

    def test_is_ready(self):
        # Test that the is_ready endpoint returns the expected values.
        #: :type: flask.wrappers.Response
        r = self.app.test_client().get('/is_ready')
        self.assertStatusCode(r, 200)
        self.assertJsonMessageRegex(r, "Yes, I'm alive.")
        self.assertIsInstance(self.app.descriptor_set, StubDescriptorSet)
        self.assertIsInstance(self.app.descriptor_generator, StubDescrGenerator)
        self.assertIsInstance(self.app.neighbor_index, StubNearestNeighborIndex)

    def test_add_descriptor_from_data_no_args(self):
        """ Test that providing no arguments causes a 400 error. """
        r = self.app.test_client().post('/add_descriptor_from_data')
        self.assertStatusCode(r, 400)

    def test_add_descriptor_from_data_no_b64(self):
        """ Test that providing no b64 data is caught + errors. """
        query_data = {
            "content_type": "text/plain"
        }
        r = self.app.test_client().post('/add_descriptor_from_data',
                                        data=query_data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "No or empty base64 data")

    def test_add_descriptor_from_data_no_content_type(self):
        query_data = {
            'data_b64': base64.b64encode(b"some test data").decode()
        }
        r = self.app.test_client().post('/add_descriptor_from_data',
                                        data=query_data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "No data mimetype")

    def test_add_descriptor_from_data_invalid_base64(self):
        """ Test sending some bad base64 data. """
        query_data = {
            'data_b64': 'not valid b64',
            'content_type': 'text/plain',
        }
        r = self.app.test_client().post('/add_descriptor_from_data',
                                        data=query_data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Failed to parse base64 data")

    def test_add_descriptor_from_data(self):
        expected_bytes = b"some test bytes"
        expected_base64 = base64.b64encode(expected_bytes).decode()
        expected_contenttype = 'text/plain'
        expected_descriptor_uid = 'test-descr-uid'
        expected_descriptor = mock.MagicMock(spec=DescriptorElement)
        expected_descriptor.uuid.return_value = expected_descriptor_uid

        self.app.describe_base64_data = mock.Mock(
            return_value=expected_descriptor
        )
        self.app.descriptor_set.add_descriptor = mock.Mock()

        query_data = {
            "data_b64": expected_base64,
            "content_type": expected_contenttype,
        }
        r = self.app.test_client().post('/add_descriptor_from_data',
                                        data=query_data)

        self.app.describe_base64_data.assert_called_once_with(
            expected_base64, expected_contenttype
        )
        self.app.descriptor_set.add_descriptor.assert_called_once_with(
            expected_descriptor
        )

        self.assertStatusCode(r, 201)
        self.assertJsonMessageRegex(r, "Success")
        r_json = json.loads(r.data.decode())
        self.assertEqual(r_json['uid'], expected_descriptor_uid)

    def test_get_nn_index_status(self):
        self.app.neighbor_index.count = mock.Mock(return_value=0)
        with self.app.test_client() as tc:
            r = tc.get('/nn_index')
            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "Success")
            self.assertEqual(json.loads(r.data.decode())['index_size'], 0)

        self.app.neighbor_index.count = mock.Mock(return_value=89756234876)
        with self.app.test_client() as tc:
            r = tc.get('/nn_index')
            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "Success")
            self.assertEqual(json.loads(r.data.decode())['index_size'],
                             89756234876)

    def test_update_nn_index_no_args(self):
        """ Test that error is returned when no arguments provided """
        r = self.app.test_client().post('/nn_index')
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "No descriptor UID JSON provided")

    def test_update_nn_index_bad_json_parse_error(self):
        """ Test handling a bad json string """
        r = self.app.test_client().post('/nn_index',
                                        data=dict(
                                            descriptor_uids='not-valid-json'
                                        ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "JSON parsing error")

    def test_update_nn_index_bad_json_not_a_list(self):
        r = self.app.test_client().post('/nn_index',
                                        data=dict(
                                            descriptor_uids='6.2'
                                        ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "JSON provided is not a list")

    def test_update_nn_index_bad_json_empty_list(self):
        r = self.app.test_client().post('/nn_index',
                                        data=dict(
                                            descriptor_uids='[]'
                                        ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "JSON list is empty")

    def test_update_nn_index_bad_json_invalid_parts(self):
        """ All UIDs provided should be hashable values. """
        r = self.app.test_client().post('/nn_index',
                                        data=dict(
                                            descriptor_uids='["a", 4, []]'
                                        ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Not all JSON list parts were hashable "
                                       "values.")

    def test_update_nn_index_uid_not_a_key(self):
        """
        Test that providing one or more UIDs that are not a part of the index
        yields an error.
        """
        def key_error_raise(*_, **__):
            raise KeyError('test-key')

        self.app.descriptor_set.get_many_descriptors = mock.Mock(
            side_effect=key_error_raise
        )
        # Pretend any UID except 2 and "hello" are not contained.
        self.app.descriptor_set.has_descriptor = mock.Mock(
            side_effect=lambda k: k == 2 or k == "hello"
        )

        expected_list = [0, 1, 2, 'hello', 'foobar']
        expected_list_json = '[0, 1, 2, "hello", "foobar"]'
        expected_bad_uids = [0, 1, 'foobar']
        r = self.app.test_client().post('/nn_index',
                                        data=dict(
                                            descriptor_uids=expected_list_json,
                                        ))
        self.app.descriptor_set.get_many_descriptors.assert_called_once_with(
            expected_list
        )
        self.app.descriptor_set.has_descriptor.assert_any_call(0)
        self.app.descriptor_set.has_descriptor.assert_any_call(1)
        self.app.descriptor_set.has_descriptor.assert_any_call(2)
        self.app.descriptor_set.has_descriptor.assert_any_call("hello")
        self.app.descriptor_set.has_descriptor.assert_any_call("foobar")
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Some provided UIDs do not exist in "
                                       "the current index")
        r_json = json.loads(r.data.decode())
        self.assertListEqual(r_json['bad_uids'], expected_bad_uids)

    def test_update_nn_index_delayed_key_error(self):
        """
        Some DescriptorSet implementations of get_many_descriptors use the
        yield statement and thus won't potentially generate key errors until
        actually iterated within the update call.  Test that this is correctly
        caught.
        """
        expected_error_key = 'expected-error-key'

        def generator_key_error(*_, **__):
            raise KeyError(expected_error_key)
            # make a generator
            # noinspection PyUnreachableCode
            yield

        self.app.descriptor_set.get_many_descriptors = mock.Mock(
            side_effect=generator_key_error
        )
        # Pretend any UID except 2 and "hello" are not contained.
        self.app.descriptor_set.has_descriptor = mock.Mock(
            side_effect=lambda k: k == 2 or k == "hello"
        )

        expected_list = [0, 1, 2, 'hello', 'foobar']
        expected_list_json = '[0, 1, 2, "hello", "foobar"]'
        expected_bad_uids = [0, 1, 'foobar']
        r = self.app.test_client().post('/nn_index',
                                        data=dict(
                                            descriptor_uids=expected_list_json,
                                        ))
        self.app.descriptor_set.get_many_descriptors.assert_called_once_with(
            expected_list
        )
        self.app.descriptor_set.has_descriptor.assert_any_call(0)
        self.app.descriptor_set.has_descriptor.assert_any_call(1)
        self.app.descriptor_set.has_descriptor.assert_any_call(2)
        self.app.descriptor_set.has_descriptor.assert_any_call("hello")
        self.app.descriptor_set.has_descriptor.assert_any_call("foobar")
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Some provided UIDs do not exist in "
                                       "the current index")
        r_json = json.loads(r.data.decode())
        self.assertListEqual(r_json['bad_uids'], expected_bad_uids)

    def test_update_nn_index(self):
        expected_descriptors = [
            mock.Mock(spec=DescriptorElement),
            mock.Mock(spec=DescriptorElement),
            mock.Mock(spec=DescriptorElement),
        ]
        expected_uid_list = [0, 1, 2]
        expected_uid_list_json = json.dumps(expected_uid_list)
        expected_new_index_count = 10

        self.app.descriptor_set.get_many_descriptors = mock.Mock(
            return_value=expected_descriptors
        )
        self.app.neighbor_index.update_index = mock.Mock()
        self.app.neighbor_index.count = mock.Mock(
            return_value=expected_new_index_count
        )

        data = dict(
            descriptor_uids=expected_uid_list_json
        )
        r = self.app.test_client().post('/nn_index', data=data)

        self.app.descriptor_set.get_many_descriptors.assert_called_once_with(
            expected_uid_list
        )
        self.app.neighbor_index.update_index.assert_called_once_with(
            expected_descriptors
        )

        self.assertStatusCode(r, 200)
        r_json = json.loads(r.data.decode())
        self.assertListEqual(r_json['descriptor_uids'], expected_uid_list)
        self.assertEqual(r_json['index_size'], expected_new_index_count)

    def test_remove_from_nn_index_no_uids(self):
        """
        Test that error is properly returned when failing to pass any UIDs to
        the endpoint.
        """
        with self.app.test_client() as tc:
            r = tc.delete('/nn_index')
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, "No descriptor UID JSON provided")

    def test_remove_from_nn_index_bad_json_parse_error(self):
        """
        Test expected error from had JSON input.
        """
        with self.app.test_client() as tc:
            r = tc.delete('/nn_index',
                          data={'descriptor_uids': 'not-valid-json'})
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, "JSON parsing error: ")

    def test_remove_from_nn_index_bad_json_not_a_list(self):
        """
        Test expected error from not sending a list.
        """
        with self.app.test_client() as tc:
            r = tc.delete('/nn_index', data={'descriptor_uids': 6.2})
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, 'JSON provided is not a list.')

    def test_remove_from_nn_index_bad_json_list_empty(self):
        """
        Test expected error when passed an empty JSON list.
        """
        with self.app.test_client() as tc:
            r = tc.delete('/nn_index',
                          data={'descriptor_uids': '[]'})
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, "JSON list is empty")

    def test_remove_from_nn_index_bad_json_invalid_parts(self):
        """
        Test expected error when input JSON contains non-hashable values.
        """
        with self.app.test_client() as tc:
            r = tc.delete('/nn_index',
                          data=dict(
                              descriptor_uids='["a", 1, []]'
                          ))
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, 'Not all JSON list parts were '
                                           'hashable values\.')

    def test_remove_from_nn_index_uid_not_a_key(self):
        """
        Test expected error when providing one or more descriptor UIDs not
        currently indexed.
        """
        expected_exception_msg = 'expected KeyError value'

        def raiseKeyError(*_, **__):
            raise KeyError(expected_exception_msg)

        # Simulate a KeyError being raised by nn-index remove call.
        self.app.neighbor_index.remove_from_index = mock.Mock(
            side_effect=raiseKeyError
        )

        with self.app.test_client() as tc:
            r = tc.delete('/nn_index',
                          data={
                              'descriptor_uids': '[0, 1, 2]'
                          })
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, "Some provided UIDs do not exist "
                                           "in the current index")
            r_json = json.loads(r.data.decode())
            self.assertListEqual(r_json['bad_uids'], [expected_exception_msg])

    def test_remove_from_nn_index(self):
        """
        Test we can remove from the nn-index via the endpoint.
        """
        self.app.neighbor_index.remove_from_index = mock.Mock()
        self.app.neighbor_index.count = mock.Mock(return_value=3)

        with self.app.test_client() as tc:
            r = tc.delete('/nn_index',
                          data=dict(
                              descriptor_uids='[1, 2, 3]'
                          ))

            self.app.neighbor_index.remove_from_index.assert_called_once_with(
                [1, 2, 3]
            )

            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "Success")
            r_json = json.loads(r.data.decode())
            self.assertListEqual(r_json['descriptor_uids'], [1, 2, 3])
            self.assertEqual(r_json['index_size'], 3)

    def test_data_nearest_neighbors_no_base64(self):
        """ Test not providing base64 """
        data = dict(
            # data_b64=base64.b64encode('test-data'),
            content_type='text/plain',
            k=10,
        )
        r = self.app.test_client().post('/data_nearest_neighbors',
                                        data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, 'No or empty base64 data provided')

    def test_data_nearest_neighbors_no_contenttype(self):
        """ Test not providing base64 """
        data = dict(
            data_b64=base64.b64encode(b'test-data').decode(),
            # content_type='text/plain',
            k=10,
        )
        r = self.app.test_client().post('/data_nearest_neighbors',
                                        data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, 'No data mimetype provided')

    def test_data_nearest_neighbors_no_k(self):
        """ Test not providing base64 """
        data = dict(
            data_b64=base64.b64encode(b'test-data').decode(),
            content_type='text/plain',
            # k=10,
        )
        r = self.app.test_client().post('/data_nearest_neighbors',
                                        data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "No 'k' value provided")

    def test_data_nearest_neighbors_bad_k_json(self):
        """ Test string for k """
        data = dict(
            data_b64=base64.b64encode(b'test-data').decode(),
            content_type='text/plain',
            k="10.2",  # float string fails an int cast.
        )
        r = self.app.test_client().post('/data_nearest_neighbors',
                                        data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Failed to convert 'k' argument to an "
                                       "integer")

    def test_data_nearest_neighbors_bad_base64(self):
        data = dict(
            data_b64="not-valid-base-64%",
            content_type='text/plain',
            k=10,
        )
        r = self.app.test_client().post('/data_nearest_neighbors',
                                        data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Failed to parse base64 data")

    def test_data_nearest_neighbors(self):
        expected_uids = ['a', 'b', 'c']
        expected_neighbors = []
        for uid in expected_uids:
            e = mock.Mock(spec=DescriptorElement)
            e.uuid.return_value = uid
            expected_neighbors.append(e)
        expected_dists = [1, 2, 3]

        self.app.describe_base64_data = mock.Mock()
        self.app.neighbor_index.nn = mock.Mock(
            return_value=[expected_neighbors, expected_dists]
        )

        data = dict(
            data_b64=base64.b64encode(b'test-data').decode(),
            content_type='text/plain',
            k=3,  # float string fails an int cast.
        )
        r = self.app.test_client().post('/data_nearest_neighbors',
                                        data=data)
        self.app.describe_base64_data.assert_called_once_with(
            data['data_b64'], data['content_type']
        )
        # noinspection PyArgumentList
        self.app.neighbor_index.nn.assert_called_once_with(
            self.app.describe_base64_data(), data['k']
        )

        self.assertStatusCode(r, 200)
        r_json = json.loads(r.data.decode())
        self.assertListEqual(r_json['neighbor_uids'], expected_uids)
        self.assertListEqual(r_json['neighbor_dists'], expected_dists)

    def test_uid_nearest_neighbors_no_uid(self):
        """ Test not providing base64 """
        data = dict(
            # uid='some-uid',
            k=10,
        )
        r = self.app.test_client().get('/uid_nearest_neighbors',
                                       data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, 'No UID provided')

    def test_uid_nearest_neighbors_no_k(self):
        """ Test not providing base64 """
        data = dict(
            uid='some-uid',
            # k=10,
        )
        r = self.app.test_client().get('/uid_nearest_neighbors',
                                       data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "No 'k' value provided")

    def test_uid_nearest_neighbors_bad_k_json(self):
        """ Test string for k """
        data = dict(
            uid='some-uid',
            k="10.2",  # float string fails an int cast.
        )
        r = self.app.test_client().get('/uid_nearest_neighbors',
                                       data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Failed to convert 'k' argument to an "
                                       "integer")

    def test_uid_nearest_neighbors_no_elem_for_uid(self):
        def raise_keyerror(*_):
            raise KeyError("invalid-key")
        self.app.descriptor_set.get_descriptor = mock.Mock(
            side_effect=raise_keyerror
        )

        data = dict(
            uid='some-uid',
            k=3,
        )
        r = self.app.test_client().get('/uid_nearest_neighbors',
                                       data=data)
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Failed to get descriptor for UID "
                                       "some-uid")

    def test_uid_nearest_neighbors(self):
        expected_uids = ['a', 'b', 'c']
        expected_neighbors = []
        for uid in expected_uids:
            e = mock.Mock(spec=DescriptorElement)
            e.uuid.return_value = uid
            expected_neighbors.append(e)
        expected_dists = [1, 2, 3]

        self.app.descriptor_set.get_descriptor = mock.Mock()
        self.app.neighbor_index.nn = mock.Mock(
            return_value=[expected_neighbors, expected_dists]
        )

        data = dict(
            uid='some-uid',
            k=10,
        )
        r = self.app.test_client().get('/uid_nearest_neighbors',
                                       data=data)
        self.app.descriptor_set.get_descriptor.assert_called_once_with(
            data['uid']
        )
        self.app.neighbor_index.nn.assert_called_once_with(
            self.app.descriptor_set.get_descriptor(), data['k']
        )

        self.assertStatusCode(r, 200)
        r_json = json.loads(r.data.decode())
        self.assertListEqual(r_json['neighbor_uids'], expected_uids)
        self.assertListEqual(r_json['neighbor_dists'], expected_dists)

    def test_get_session_info_no_session_id(self):
        """
        Test that passing no session ID results in a 400 error.
        """
        self._test_getter_no_sid('session')

    def test_get_session_info_invalid_session_id(self):
        """
        Test that passing an ID that does not map to any current session
        returns a 400 error.
        """
        # There are no sessions on server initialization.
        self._test_getter_sid_not_found('session')

        iqrs = IqrSession(session_uid='1')  # not '0', which is queried for.
        self.app.controller.add_session(iqrs)
        self._test_getter_sid_not_found('session')

    def test_get_session_info(self):
        """
        Test a valid retrieval of a complex IQR session state.
        """
        iqrs = IqrSession(session_uid='abc')

        ep, en, p1, p2, p3, n1, n2, d1, d2, n3 = [
            DescriptorMemoryElement('test', uid) for uid in
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            # ep   en   p1   p2   p3   n1   n2   d1   d2   n3
        ]   # C              C         C    C    C    C
        #     ^Contributing^

        # Current adjudications
        iqrs.external_positive_descriptors = {ep}
        iqrs.positive_descriptors = {p1, p2, p3}
        iqrs.external_negative_descriptors = {en}
        iqrs.negative_descriptors = {n1, n2, n3}
        # "Last Refine" adjudications
        # - simulating that "currently" neutral descriptors were previous
        #   adjudicated.
        iqrs.rank_contrib_pos = {p2, d1}
        iqrs.rank_contrib_pos_ext = {ep}
        iqrs.rank_contrib_neg = {n1, n3, d2}
        iqrs.rank_contrib_neg_ext = set()
        # mock working set with
        iqrs.working_set.add_many_descriptors([p1, p2, p3, n1, n2, d1, d2, n3])

        self.app.controller.add_session(iqrs)

        with self.app.test_client() as tc:
            #: :type: flask.wrappers.Response
            r = tc.get('/session?sid=abc')
            self.assertStatusCode(r, 200)
            r_json = r.json
            assert r_json['sid'] == 'abc'
            # That everything included in "current" adjudications is included
            # here.
            assert set(r_json['uuids_pos_ext']) == {'a'}
            assert set(r_json['uuids_pos']) == {'c', 'd', 'e'}
            assert set(r_json['uuids_neg_ext']) == {'b'}
            assert set(r_json['uuids_neg']) == {'f', 'g', 'j'}
            # That those marked as "contributing" are included here
            assert set(r_json['uuids_pos_in_model']) == {'d', 'h'}
            assert set(r_json['uuids_pos_ext_in_model']) == {'a'}
            assert set(r_json['uuids_neg_in_model']) == {'f', 'j', 'i'}
            assert set(r_json['uuids_neg_ext_in_model']) == set()
            # IQR working set expected size
            assert r_json['wi_count'] == 8

    def test_refine_no_session_id(self):
        with self.app.test_client() as tc:
            r = tc.post('/refine')
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, "No session id \(sid\) provided")

    def test_refine_invalid_session_id(self):
        with self.app.test_client() as tc:
            r = tc.post('/refine',
                        data={
                            'sid': 'invalid-sid'
                        })
            self.assertStatusCode(r, 404)
            self.assertJsonMessageRegex(r, "session id invalid-sid not found")

    def _test_getter_no_sid(self, endpoint):
        """
        Test common getter response to providing no session ID.
        :param str endpoint: String endpoint of the getter method.
        """
        with self.app.test_client() as tc:
            r = tc.get('/{}'.format(endpoint))
            self.assertStatusCode(r, 400)
            self.assertJsonMessageRegex(r, r"No session id \(sid\) provided")

    def _test_getter_sid_not_found(self, endpoint):
        """
        Test common getter response to when the given session ID is not present
        in the controller.
        :param str endpoint: String endpoint of the getter method.
        """
        with self.app.test_client() as tc:
            # Service IQR session controller is empty upon construction, so no
            # ID is initially valid.
            r = tc.get('/{}?sid=0'.format(endpoint))
            self.assertStatusCode(r, 404)
            self.assertJsonMessageRegex(r, "session id '0' not found")

    def test_get_results_no_sid(self):
        """
        Test getting relevancy results without providing a session ID.
        """
        self._test_getter_no_sid('get_results')

    def test_get_results_sid_not_found(self):
        """
        Test that the expected error is returned when the given session ID is
        not present in the controller.
        """
        self._test_getter_sid_not_found('get_results')

    def test_get_results(self):
        """
        Test successfully getting results from a requested session.
        """
        # Mock controller interaction to get a mock IqrSession instance.
        self.app.controller.has_session_uuid = \
            mock.MagicMock(return_value=True)
        self.app.controller.get_session = mock.MagicMock()
        # Mock IQR session instance to have
        # Mock results return to be something valid.
        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        self.app.controller.get_session().ordered_results.return_value = [
            [d0, 0.3], [d2, 0.2], [d1, 0.1],
        ]

        test_sid = '0000'
        with self.app.test_client() as tc:
            r = tc.get('/get_results?sid={}'.format(test_sid))
            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "Returning result pairs")
            r_json = r.json
            assert r_json['total_results'] == 3
            assert r_json['results'] == [[0, 0.3], [2, 0.2], [1, 0.1]]

        self.app.controller.has_session_uuid.assert_called_once_with(test_sid)

    def test_get_positive_adjudication_relevancy_no_sid(self):
        """
        Test that the expected error is returned when no session ID is
        provided.
        """
        self._test_getter_no_sid('get_positive_adjudication_relevancy')

    def test_get_positive_adjudication_relevancy_sid_not_found(self):
        """
        Test that the expected error is returned when the given session ID is
        not present in the controller.
        """
        self._test_getter_sid_not_found('get_positive_adjudication_relevancy')

    def test_get_positive_adjudication_relevancy(self):
        """
        Test successfully getting results for descriptors that are positively
        adjudicated.
        """
        # Mock controller interaction to get a mock IqrSession instance.
        self.app.controller.has_session_uuid = \
            mock.MagicMock(return_value=True)
        self.app.controller.get_session = mock.MagicMock()
        # Mock IQR session instance to have
        # Mock results return to be something valid.
        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        self.app.controller.get_session().get_positive_adjudication_relevancy \
            .return_value = [
                [d0, 0.3], [d2, 0.2], [d1, 0.1],
            ]

        test_sid = '0000'
        with self.app.test_client() as tc:
            r = tc.get('/get_positive_adjudication_relevancy?sid={}'
                       .format(test_sid))
            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "success")
            r_json = r.json
            assert r_json['total'] == 3
            assert r_json['results'] == [[0, 0.3], [2, 0.2], [1, 0.1]]

        self.app.controller.has_session_uuid.assert_called_once_with(test_sid)

    def test_get_negative_adjudication_relevancy_no_sid(self):
        """
        Test that the expected error is returned when no session ID is
        provided.
        """
        self._test_getter_no_sid('get_negative_adjudication_relevancy')

    def test_get_negative_adjudication_relevancy_sid_not_found(self):
        """
        Test that the expected error is returned when the given session ID is
        not present in the controller.
        """
        self._test_getter_sid_not_found('get_negative_adjudication_relevancy')

    def test_get_negative_adjudication_relevancy(self):
        """
        Test successfully getting results for descriptors that are positively
        adjudicated.
        """
        # Mock controller interaction to get a mock IqrSession instance.
        self.app.controller.has_session_uuid = \
            mock.MagicMock(return_value=True)
        self.app.controller.get_session = mock.MagicMock()
        # Mock IQR session instance to have
        # Mock results return to be something valid.
        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        self.app.controller.get_session().get_negative_adjudication_relevancy \
            .return_value = [
                [d0, 0.3], [d2, 0.2], [d1, 0.1],
            ]

        test_sid = '0000'
        with self.app.test_client() as tc:
            r = tc.get('/get_negative_adjudication_relevancy?sid={}'
                       .format(test_sid))
            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "success")
            r_json = r.json
            assert r_json['total'] == 3
            assert r_json['results'] == [[0, 0.3], [2, 0.2], [1, 0.1]]

        self.app.controller.has_session_uuid.assert_called_once_with(test_sid)

    def test_get_unadjudicated_relevancy_no_sid(self):
        """
        Test that the expected error is returned when no session ID is
        provided.
        """
        self._test_getter_no_sid('get_unadjudicated_relevancy')

    def test_get_unadjudicated_relevancy_sid_not_found(self):
        """
        Test that the expected error is returned when the given session ID is
        not present in the controller.
        """
        self._test_getter_sid_not_found('get_unadjudicated_relevancy')

    def test_get_unadjudicated_relevancy(self):
        """
        Test successfully getting results for descriptors that are positively
        adjudicated.
        """
        # Mock controller interaction to get a mock IqrSession instance.
        self.app.controller.has_session_uuid = \
            mock.MagicMock(return_value=True)
        self.app.controller.get_session = mock.MagicMock()
        # Mock IQR session instance to have
        # Mock results return to be something valid.
        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        self.app.controller.get_session().get_unadjudicated_relevancy \
            .return_value = [
                [d0, 0.3], [d2, 0.2], [d1, 0.1],
            ]

        test_sid = '0000'
        with self.app.test_client() as tc:
            r = tc.get('/get_unadjudicated_relevancy?sid={}'
                       .format(test_sid))
            self.assertStatusCode(r, 200)
            self.assertJsonMessageRegex(r, "success")
            r_json = r.json
            assert r_json['total'] == 3
            assert r_json['results'] == [[0, 0.3], [2, 0.2], [1, 0.1]]

        self.app.controller.has_session_uuid.assert_called_once_with(test_sid)

    @mock.patch('smqtk.web.iqr_service.iqr_server.SupervisedClassifier'
                '.get_impls')
    def test_classify(self, m_sc_get_impls):
        """
        Test calling the GET /classify endpoint under nominal conditions.
        """
        # Mock classifier instantiation to return stub classifier class
        # instance.
        m_sc_get_impls.return_value = {StubClassifier}

        # Setup descriptor set to have descriptors we test query for
        mock_descriptors = [
            DescriptorMemoryElement('', 'a').set_vector([0.4]),
            DescriptorMemoryElement('', 'b').set_vector([0.5]),
            DescriptorMemoryElement('', 'c').set_vector([0.6]),
        ]
        self.app.descriptor_set.get_many_descriptors = mock.MagicMock(
            return_value=mock_descriptors
        )
        # Mock stub classifier return
        mock_classifications = [
            MemoryClassificationElement('', 'a'),
            MemoryClassificationElement('', 'b'),
            MemoryClassificationElement('', 'c'),
        ]
        mock_classifications[0].set_classification(
            {'positive': 0.6, 'negative': 0.4})
        mock_classifications[1].set_classification(
            {'positive': 0.5, 'negative': 0.5})
        mock_classifications[2].set_classification(
            {'positive': 0.4, 'negative': 0.6})
        StubClassifier.classify_elements = mock.MagicMock(
            return_value=iter(mock_classifications)
        )

        with self.app.test_client() as tc:
            # Initialize a new session.
            tc.post("/session",
                    data=dict(
                        sid="0"
                    ))

            # Setup nominal IQR Session state for classifier generation and
            # application.
            iqr_session = self.app.controller.get_session("0")
            # Mock descriptors on IQR session
            iqr_session.external_positive_descriptors.add(
                DescriptorMemoryElement('', 0).set_vector([0.0])
            )
            iqr_session.positive_descriptors.add(
                DescriptorMemoryElement('', 1).set_vector([0.1])
            )
            iqr_session.external_negative_descriptors.add(
                DescriptorMemoryElement('', 2).set_vector([1.0])
            )
            iqr_session.negative_descriptors.add(
                DescriptorMemoryElement('', 3).set_vector([0.9])
            )

            #: :type: flask.Response
            r = tc.get('/classify',
                       query_string=dict(
                           sid=0,
                           uuids=json.dumps(['a', 'b', 'c'])
                       ))
            self.assertStatusCode(r, 200)
            # We expect the UIDs returned to be in the same order as input and
            # for the expected classification "positive" probabilities as in
            # the mocked classification results.
            r_json = r.json
            assert r_json['sid'] == '0'
            assert r_json['uuids'] == ['a', 'b', 'c']
            assert r_json['proba'] == [0.6, 0.5, 0.4]

    def test_get_iqr_state_no_sid(self):
        # Test that calling GET /state with no SID results in error.
        r = self.app.test_client().get('/state')
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, 'No session id')

    def test_get_iqr_state_bad_sid_empty_controller(self):
        # Test that giving an invalid SID to get_iqr_state results in error
        # message return, when no sessions in controller.

        # App should currently have no sessions.
        self.assertTupleEqual(
            self.app.controller.session_uuids(),
            ()
        )

        # Attempt getting a state with an SID not currently registered in the
        # controller.
        r = self.app.test_client().get('/state',
                                       query_string=dict(
                                           sid='something-invalid'
                                       ))
        self.assertStatusCode(r, 404)
        self.assertJsonMessageRegex(r, 'not found')

    def test_get_iqr_state_base_sid_nonempty_controller(self):
        # Test that giving an invalid SID to get_iqr_state results in error
        # message return when controller has one or more sessions in it.

        # Initialize two different sessions
        self.app.test_client().post('/session',
                                    data=dict(
                                        sid='test-session-1'
                                    ))
        self.app.test_client().post('/session',
                                    data=dict(
                                        sid='test-session-2'
                                    ))

        # attempt to get state for a session not created
        r = self.app.test_client().get('/state',
                                       query_string=dict(
                                           sid='test-session-3'
                                       ))
        self.assertStatusCode(r, 404)
        self.assertJsonMessageRegex(r, 'not found')

    def test_get_iqr_state(self):
        # Test that the base64 returned is what is returned from
        #  IqrSession.get_state_bytes (mocked)
        expected_bytes = b'these-bytes'
        expected_b64 = base64.b64encode(expected_bytes).decode()

        # Pretend input SID is valid
        self.app.controller.has_session_uuid = mock.Mock(return_value=True)
        self.app.controller.get_session = mock.Mock()
        self.app.controller.get_session().get_state_bytes.return_value = \
            expected_bytes

        r = self.app.test_client().get('/state',
                                       query_string=dict(
                                           sid='some-sid'
                                       ))
        self.assertStatusCode(r, 200)
        r_json = json.loads(r.data.decode())
        self.assertEqual(r_json['message'], "Success")
        self.assertEqual(r_json['sid'], 'some-sid')
        self.assertEqual(r_json['state_b64'], expected_b64)

    def test_set_iqr_state_no_sid(self):
        # Test that calling set_iqr_state with no SID returns an error
        r = self.app.test_client().put('/state',
                                       data=dict(
                                           state_base64='dummy'
                                       ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "No session id \(sid\) provided")

    def test_set_iqr_state_no_b64(self):
        """
        Test that calling set_iqr_state with no base_64 data returns an
        error.
        """
        r = self.app.test_client().put('/state',
                                       data=dict(
                                           sid='dummy'
                                       ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, 'No state package base64 provided')

    def test_set_iqr_state_invalid_base64(self):
        # Test when PUT /state is given invalid base64 data.

        r = self.app.test_client().put('/state',
                                       data=dict(
                                           sid='test-sid',
                                           state_base64='some-invalid-data'
                                       ))
        self.assertStatusCode(r, 400)
        self.assertJsonMessageRegex(r, "Invalid base64 input")

    def test_set_iqr_state_invalid_sid(self):
        # Test that an invalid SID provided causes an error.  Must set a state
        # to an existing session.
        test_b64 = base64.b64encode(b'test')

        # App starts with nothing in session controller, so no SID should be
        # initially valid.
        r = self.app.test_client().put('/state',
                                       data=dict(
                                           sid='invalid',
                                           state_base64=test_b64
                                       ))
        self.assertStatusCode(r, 404)
        self.assertJsonMessageRegex(r, 'not found')

    def test_set_iqr_state(self):
        # Test that expected base64 decoded bytes are passed to
        # `IqrSession.set_state_bytes` method.
        expected_sid = 'test-sid'
        expected_bytes = b'expected-state-bytes'
        expected_bytes_b64 = base64.b64encode(expected_bytes).decode()

        self.app.controller.has_session_uuid = mock.Mock(return_value=True)
        self.app.controller.get_session = mock.Mock()

        r = self.app.test_client().put('/state',
                                       data=dict(
                                           sid=expected_sid,
                                           state_base64=expected_bytes_b64
                                       ))

        self.app.controller.get_session().set_state_bytes.assert_called_with(
            expected_bytes, self.app.descriptor_factory
        )
        self.assertStatusCode(r, 200)
        r_json = json.loads(r.data.decode())
        self.assertRegex(r_json['message'], 'Success')
        self.assertRegex(r_json['sid'], expected_sid)

    def test_get_random_uids(self):
        """
        Test getting random descriptor UIDs from the
        """
        expected = list(map(chr, range(97, 97+26)))
        self.app.descriptor_set.iterkeys = mock.MagicMock(
            return_value=list(map(chr, range(97, 97+26)))
        )
        with self.app.test_client() as tc:
            r = tc.get('/random_uids')
            self.assertStatusCode(r, 200)
            r_json = r.json
            assert r_json['total'] == 26
            # The results should have the expected contents but not be in the
            # same order, cause ya know, random.
            assert sorted(r_json['results']) == expected
            assert r_json['results'] != expected
            result1 = r_json['results']

            # A second call should return the same list due to caching
            r2 = tc.get('/random_uids')
            self.assertStatusCode(r, 200)
            r2_json = r2.json
            assert r2_json['results'] == result1

            # Calling with refresh should re-query the descriptor set and
            # reorder. Result (should) be different.
            # - MAYBE fails on RARE occasions because shuffle resulted in a
            #   duplicate ordering? I would hope not given pseudo-randomness
            #   but I don't know.
            r3 = tc.get('/random_uids?refresh=true')
            self.assertStatusCode(r, 200)
            r3_json = r3.json
            assert r3_json['results'] != result1

    def test_get_random_uids_paged(self):
        """ Test pagination of random UIDs """
        expected = list(map(chr, range(97, 97+26)))
        self.app.descriptor_set.iterkeys = mock.MagicMock(
            return_value=list(map(chr, range(97, 97+26)))
        )
        with self.app.test_client() as tc:
            # Lets get a baseline to test pagination
            rbase = tc.get('/random_uids')
            self.assertStatusCode(rbase, 200)
            result_all = rbase.json['results']

            r = tc.get('/random_uids?i=3')
            self.assertStatusCode(r, 200)
            assert r.json['results'] == result_all[3:]
            
            r = tc.get('/random_uids?j=-4')
            self.assertStatusCode(r, 200)
            assert r.json['results'] == result_all[:-4]

            r = tc.get('/random_uids?j=10')
            self.assertStatusCode(r, 200)
            assert r.json['results'] == result_all[:10]

            r = tc.get('/random_uids?i=7&j=10')
            self.assertStatusCode(r, 200)
            assert r.json['results'] == result_all[7:10]
