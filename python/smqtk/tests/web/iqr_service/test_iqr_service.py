import base64
import json
import mock
import os
import unittest

from smqtk.algorithms import Classifier, DescriptorGenerator, \
    NearestNeighborsIndex
from smqtk.web.iqr_service import IqrService


THIS_MODULE_PATH = 'smqtk.tests.web.iqr_service.test_iqr_service'


class StubClassifier (Classifier):
    """
    Classifier stub for testing IqrService.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def get_labels(self):
        pass

    def _classify(self, d):
        pass


class StubDescrGenerator (DescriptorGenerator):
    """
    DescriptorGenerator stub for testing IqrService.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def valid_content_types(self):
        pass

    def _compute_descriptor(self, data):
        pass


class StubNearestNeighborIndex (NearestNeighborsIndex):
    """
    NearestNeighborIndex stub for testing IqrService.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def count(self):
        pass

    def _build_index(self, descriptors):
        pass

    def _update_index(self, descriptors):
        pass

    def _nn(self, d, n=1):
        pass


class TestIqrService (unittest.TestCase):

    # Patch in this module for stub implementation access.
    # noinspection PyUnresolvedReferences
    @mock.patch.dict(os.environ, {
        "CLASSIFIER_PATH": THIS_MODULE_PATH,
        "DESCRIPTOR_GENERATOR_PATH": THIS_MODULE_PATH,
        "NN_INDEX_PATH": THIS_MODULE_PATH,
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
        plugin_config['descriptor_index']['type'] = 'MemoryDescriptorIndex'

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
        self.assertEqual(r.status_code, code)

    def assertJsonMessageRegex(self, r, regex):
        """
        :type r: flask.wrappers.Response
        :type regex: str
        """
        self.assertRegexpMatches(json.loads(r.data)['message'], regex)

    # Test Methods #############################################################

    def test_is_ready(self):
        # Test that the is_ready endpoint returns the expected values.
        #: :type: flask.wrappers.Response
        r = self.app.test_client().get('/is_ready')
        self.assertStatusCode(r, 200)
        self.assertJsonMessageRegex(r, "Yes, I'm alive.")

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
        expected_bytes = 'these-bytes'
        expected_b64 = base64.b64encode(expected_bytes.encode('utf8'))

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
        r_json = json.loads(r.data)
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
        # Test that calling set_iqr_state with no base_64 data returns an error.
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
        test_b64 = base64.b64encode('test'.encode('utf8'))

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
        expected_bytes = 'expected-state-bytes'
        expected_bytes_b64 = base64.b64encode(expected_bytes.encode('utf8'))

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
        r_json = json.loads(r.data)
        self.assertRegexpMatches(r_json['message'], 'Success')
        self.assertRegexpMatches(r_json['sid'], expected_sid)
