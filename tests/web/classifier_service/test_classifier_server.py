import base64
import json
import math
import unittest.mock as mock
import os
import unittest

from six.moves import cPickle as pickle

from smqtk.algorithms import Classifier, RankRelevancy
from smqtk.iqr import IqrSession
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.utils.plugin import Pluggable, OS_ENV_PATH_SEP
from smqtk.web.classifier_service.classifier_server import \
    SmqtkClassifierService

from .dummy_classifier import DummyClassifier, DummySupervisedClassifier, \
    STUB_CLASSIFIER_MOD_PATH


class TestClassifierService (unittest.TestCase):

    # noinspection PyUnresolvedReferences
    @mock.patch.dict(os.environ, {
        Pluggable.PLUGIN_ENV_VAR:
            OS_ENV_PATH_SEP.join([
                STUB_CLASSIFIER_MOD_PATH,
                'tests.web.classifier_service.dummy_descriptor_generator',
            ])
    })
    def setUp(self):
        super(TestClassifierService, self).setUp()
        self.config = SmqtkClassifierService.get_default_config()

        key_mce = "smqtk.representation.classification_element.memory.MemoryClassificationElement"
        key_c_dummy = "tests.web.classifier_service.dummy_classifier.DummyClassifier"
        key_dme = "smqtk.representation.descriptor_element.local_elements.DescriptorMemoryElement"
        key_dg_dummy = "tests.web.classifier_service.dummy_descriptor_generator.DummyDescriptorGenerator"
        key_sc_dummy = "tests.web.classifier_service.dummy_classifier.DummySupervisedClassifier"

        self.config['classification_factory']['type'] = key_mce
        # del self.config['classification_factory'][
        #     'smqtk.representation.classification_element.file'
        #     '.FileClassificationElement'
        # ]

        del self.config['classifier_collection']['__example_label__']
        self.dummy_label = 'dummy'
        self.config['classifier_collection'][self.dummy_label] = {
            key_c_dummy: {},
            'type': key_c_dummy
        }
        self.config['immutable_labels'] = [self.dummy_label]

        self.config['descriptor_factory']['type'] = key_dme
        # del self.config['descriptor_factory']['DescriptorFileElement']

        self.config['descriptor_generator'] = {
            key_dg_dummy: {},
            'type': key_dg_dummy
        }

        self.config['iqr_state_classifier_config']['type'] = key_sc_dummy

        self.config['enable_classifier_removal'] = True

        self.config['flask_app'] = {}
        del self.config['server']

        self.app = SmqtkClassifierService(json_config=self.config)

    def assertStatus(self, rv, code):
        self.assertEqual(rv.status_code, code)

    def assertResponseMessageRegex(self, rv, regex):
        self.assertRegex(json.loads(rv.data.decode())['message'], regex)

    def assertMessage(self, resp_data, message):
        self.assertEqual(message, resp_data['message'])

    def test_server_alive(self):
        rv = self.app.test_client().get('/is_ready')

        self.assertStatus(rv, 200)
        resp_data = json.loads(rv.data.decode())
        self.assertMessage(resp_data, "Yes, I'm alive!")

    def test_preconfigured_labels(self):
        rv = self.app.test_client().get('/classifier_labels')

        self.assertStatus(rv, 200)
        resp_data = json.loads(rv.data.decode())
        self.assertMessage(resp_data, "Classifier labels.")
        self.assertListEqual(resp_data['labels'], ['dummy'])

    def test_one_classify(self):
        results_exp = dict(positive=0.5, negative=0.5)
        label = 'dummy'
        content_type = 'text/plain'
        element = base64.b64encode(b'TEST ELEMENT').decode()

        with self.app.test_client() as cli:
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Finished classification.")
            self.assertDictEqual(resp_data['result'][label], results_exp)

            # Same classifier, just retrieving it different ways, so skip the
            # correctness check
            rv = cli.post('/classify', data={
                'label': label,
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)

            rv = cli.post('/classify', data={
                'label': json.dumps(label),
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)

            rv = cli.post('/classify', data={
                'label': json.dumps([label]),
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)

    def test_adjusted_classify(self):
        results_exp = dict(positive=0.5, negative=0.5)
        label = 'dummy'
        content_type = 'text/plain'
        element = base64.b64encode(b'TEST ELEMENT').decode()

        with self.app.test_client() as cli:
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
                'adjustment': json.dumps({
                    'positive': 0,
                }),
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Finished classification.")
            self.assertDictEqual(resp_data['result'][label], results_exp)

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
                'adjustment': json.dumps({
                    'positive': -1,
                }),
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Finished classification.")
            result = resp_data['result'][label]
            self.assertAlmostEqual(result['positive'], 1/(1+math.exp(-1)))
            self.assertAlmostEqual(result['negative'], 1/(1+math.exp(1)))

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
                'adjustment': json.dumps({
                    'positive': 1,
                }),
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Finished classification.")
            result = resp_data['result'][label]
            self.assertAlmostEqual(result['positive'], 1/(1+math.exp(1)))
            self.assertAlmostEqual(result['negative'], 1/(1+math.exp(-1)))

    def test_multiple_classify(self):
        content_type = 'text/plain'
        element = base64.b64encode(b'TEST ELEMENT').decode()
        results_exp = dict(positive=0.5, negative=0.5)
        pickle_data = pickle.dumps(DummyClassifier.from_config({}))
        enc_data = base64.b64encode(pickle_data)
        old_label = 'dummy'
        new_label = 'dummy2'
        lock_clfr_str = 'true'

        with self.app.test_client() as cli:
            rv = cli.post('/classifier', data={
                'label': new_label,
                'lock_label': lock_clfr_str,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 201)
            resp_data = json.loads(rv.data.decode())
            self.assertEqual(resp_data["message"],
                             "Uploaded classifier for label '%s'."
                             % new_label)
            self.assertEqual(resp_data["label"], new_label)

            rv = cli.post('/classify', data={
                'label': new_label,
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Finished classification.")
            self.assertDictEqual(resp_data['result'][new_label], results_exp)

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Finished classification.")
            self.assertDictEqual(resp_data['result'][old_label], results_exp)
            self.assertDictEqual(resp_data['result'][new_label], results_exp)

    def test_get_add_del_classifier(self):
        old_label = 'dummy'
        new_label = 'dummy2'

        with self.app.test_client() as cli:
            rv = cli.get(f'/classifier?label={old_label}')
            self.assertStatus(rv, 200)
            enc_data = rv.data.decode()

            rv = cli.post('/classifier', data={
                'label': new_label,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 201)
            resp_data = json.loads(rv.data.decode())
            self.assertEqual(resp_data["message"],
                             "Uploaded classifier for label '%s'."
                             % new_label)
            self.assertEqual(resp_data["label"], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']),
                                {old_label, new_label})

            rv = cli.delete('/classifier', data={
                'label': new_label,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertEqual(resp_data["message"],
                             "Removed classifier with label '%s'."
                             % new_label)
            self.assertEqual(resp_data["removed_label"], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']), {old_label})

    def test_add_imm_del_classifier(self):
        pickle_data = pickle.dumps(DummyClassifier.from_config({}))
        enc_data = base64.b64encode(pickle_data).decode('utf8')
        old_label = 'dummy'
        new_label = 'dummy2'
        lock_clfr_str = 'true'

        with self.app.test_client() as cli:
            rv = cli.post('/classifier', data={
                'label': new_label,
                'lock_label': lock_clfr_str,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 201)
            resp_data = json.loads(rv.data.decode())
            self.assertEqual(resp_data["message"],
                             "Uploaded classifier for label '%s'."
                             % new_label)
            self.assertEqual(resp_data["label"], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']),
                                {old_label, new_label})

            rv = cli.delete('/classifier', data={
                'label': new_label,
            })
            self.assertStatus(rv, 405)
            resp_data = json.loads(rv.data.decode())
            self.assertEqual(resp_data["message"],
                             "Label '%s' refers to a classifier that is"
                             " immutable." % new_label)
            self.assertEqual(resp_data['label'], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']),
                                {old_label, new_label})

    def test_post_classifier_failures(self):
        pickle_data = pickle.dumps(DummyClassifier.from_config({}))
        enc_data = base64.b64encode(pickle_data)
        bad_data = base64.b64encode(pickle.dumps(object()))
        old_label = 'dummy'
        new_label = 'dummy2'
        lock_clfr_str = '['

        with self.app.test_client() as cli:
            rv = cli.post('/classifier', data={
                'label': old_label,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 400)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data,
                               "Label '%s' already exists in classifier"
                               " collection." % old_label)
            self.assertEqual(resp_data['label'], old_label)

            rv = cli.post('/classifier', data={'label': old_label})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "No state base64 data provided.")

            rv = cli.post('/classifier', data={'bytes_b64': enc_data})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "No descriptive label provided.")

            rv = cli.post('/classifier', data={
                'label': old_label,
                'lock_label': lock_clfr_str,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "Invalid boolean value for 'lock_label'."
                               " Was given: '%s'" % lock_clfr_str)

            rv = cli.post('/classifier', data={
                'label': new_label,
                'bytes_b64': bad_data,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "Data added for label '%s' is not a"
                               " Classifier." % new_label)

    def test_del_classifier_failures(self):
        old_label = 'dummy'
        new_label = 'dummy2'

        with self.app.test_client() as cli:
            rv = cli.delete('/classifier', data={})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "No label provided.")

            rv = cli.delete('/classifier', data={'label': old_label})
            self.assertStatus(rv, 405)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data,
                               "Label '%s' refers to a classifier that is"
                               " immutable." % old_label)
            self.assertEqual(resp_data['label'], old_label)

            rv = cli.delete('/classifier', data={'label': new_label})
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data,
                               "Label '%s' does not refer to a classifier"
                               " currently registered." % new_label)
            self.assertEqual(resp_data['label'], new_label)

    def test_get_classifier_failures(self):
        label = 'dummy2'

        with self.app.test_client() as cli:
            rv = cli.get('/classifier', data={})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "No label provided.")

            rv = cli.get(f'/classifier?label={label}')
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data.decode())
            self.assertMessage(resp_data,
                               "Label '%s' does not refer to a classifier"
                               " currently registered." % label)
            self.assertEqual(resp_data['label'], label)

    def test_classify_uids_no_classifiers(self):
        """ Test that an empty result response comes back when no classifiers
        are registered in the service. """
        # Empty out the classifier collection for the purpose of this test.
        for la in self.app.classifier_collection.labels():
            self.app.classifier_collection.remove_classifier(la)

        with self.app.test_client() as cli:
            rv = cli.post("classify_uids", data=dict(
                uid_list=[0, 1, 2],  # just some meaningless UIDs
            ))
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data,
                               "No classifiers currently loaded.")
            assert resp_data['result'] == {}

    def test_classify_uids_no_uid_list(self):
        """ Test the appropriate error is returned when no UID list is
        provided."""
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids")
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json, "No UIDs provided.")

    def test_classify_uids_empty_uid_list(self):
        """ Test the appropriate error is returned when an empty list is
        provided. """
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list='[]'
            ))
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json, "No UIDs provided.")

    def test_classify_uids_bad_uid_json(self):
        """ Test that the appropriate error is returned when invalid JSON is
        provided. """
        uid_list_json = 'not json'
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=uid_list_json
            ))
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json, "Failed to parse JSON list of UIDs.")

    def test_classify_uids_invalid_label_json(self):
        """ Test providing labels but with a value that is invalid json and
        that the appropriate error occurs. """
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=json.dumps(['a', 'b', 'c']),
                label="[",
            ))
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json,
                               "Invalid label(s) specified: "
                               "Label is not a properly formatted JSON nor a "
                               "simple string.")

    def test_classify_uids_invalid_label_json_value(self):
        """ Test providing labels that is a valid json but not an acceptable
        type and that the appropriate error occurs. """
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=json.dumps(['a', 'b', 'c']),
                label="{}",
            ))
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json,
                               "Invalid label(s) specified: "
                               "Label must be a list of strings or a single "
                               "string (given type: dict).")

    def test_classify_uids_invalid_label_json_inner_value(self):
        """ Test providing a valid json string but with an inner value that is
        not a string and that the appropriate error is returned. """
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=json.dumps(['a', 'b', 'c']),
                label='["label", []]',
            ))
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json,
                               "Invalid label(s) specified: "
                               "Label must be a list of strings or a single "
                               "string: give a list of more than just strings "
                               "(found type: list).")

    def test_classify_uids_missing_labels(self):
        """ Test providing a label for a classifier that is not present in the
        service and that the appropriate error is returned. This check is
        expected to occur before attempting descriptor retrieval or
        classification. """
        missing_clfrs = ['not-present', 'dummy']
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=json.dumps(['a', 'b', 'c']),
                label=json.dumps(missing_clfrs),
            ))
            self.assertStatus(rv, 404)
            rv_json = rv.json
            # noinspection PyUnresolvedReferences
            assert rv_json['message'].startswith(
                "The following labels are not registered with any "
                "classifiers: "
            )
            # noinspection PyUnresolvedReferences
            assert set(rv_json['missing_labels']) == \
                (set(missing_clfrs) - {"dummy"})

    def test_classify_uids_missing_uids(self):
        """ Test that the appropriate error occurs when requesting descriptor
        UIDs that are not present in the configured descriptor set. """
        # Default setup construction specifies a default, empty descriptor set
        # so any UIDs will be "missing".
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=json.dumps(['a', 'b', 'c']),
            ))
            self.assertStatus(rv, 400)
            # noinspection PyTypeChecker
            self.assertMessage(rv.json, "One or more input UIDs did not exist "
                                        "in the configured descriptor set!")

    def test_classify_uids(self):
        """ Test a simple invocation. """
        # Add a single descriptor to the default empty DescriptorSet.
        self.app.descriptor_set.add_descriptor(
            DescriptorMemoryElement('test', 0).set_vector([0]))
        with self.app.test_client() as cli:
            #: :type: requests.Response
            rv = cli.post("classify_uids", data=dict(
                uid_list=json.dumps([0]),
            ))
            self.assertStatus(rv, 200)
            rv_json = rv.json
            # DummyClassifier impl is known to just return 50/50
            # classifications.
            # noinspection PyUnresolvedReferences
            assert rv_json['result'] == {
                "dummy": [{"positive": 0.5, "negative": 0.5}]
            }

    def test_classify_failures(self):
        content_type = 'text/plain'
        bytes_b64 = base64.b64encode(b'TEST ELEMENT').decode()
        label_invalid_json_failure = '['
        label_valid_json_failure = '{}'
        label_valid_json_list_failure = '["test", {}]'
        missing_clfrs_1 = ['dummy', 'foo']
        missing_clfrs_2 = ['dummy', 'foo', 'bar']
        missing_clfrs_3 = ['foo']
        missing_clfrs_4 = ['foo', 'bar']

        with self.app.test_client() as cli:
            # When we provide no base-64 data
            rv = cli.post('/classify', data={
                'content_type': content_type,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "No base-64 bytes provided.")

            # When we provide no content type for the data provided.
            rv = cli.post('/classify', data={
                'bytes_b64': bytes_b64,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "No content type provided.")

            # When we provide invalid JSON for the labels.
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': label_invalid_json_failure,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "Invalid label(s) specified: "
                               "Label is not a properly formatted JSON nor a "
                               "simple string.")

            # When we provide valid json, but neither a string nor list
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': label_valid_json_failure,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "Invalid label(s) specified: "
                               "Label must be a list of strings or a single "
                               "string (given type: dict).")

            # When one value of a list of labels is not a string.
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': label_valid_json_list_failure,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data.decode()),
                               "Invalid label(s) specified: "
                               "Label must be a list of strings or a "
                               "single string: give a list of more "
                               "than just strings (found type: dict).")

            # When providing labels for classifiers not present in the service.
            # 4 variations.
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': json.dumps(missing_clfrs_1),
            })
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data.decode())
            self.assertTrue(resp_data['message'].startswith(
                "The following labels are not registered with any"
                " classifiers:"))
            self.assertSetEqual(set(resp_data['missing_labels']),
                                set(missing_clfrs_1) - {'dummy'})

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': json.dumps(missing_clfrs_2),
            })
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data.decode())
            self.assertTrue(resp_data['message'].startswith(
                "The following labels are not registered with any"
                " classifiers:"))
            self.assertSetEqual(set(resp_data['missing_labels']),
                                set(missing_clfrs_2) - {'dummy'})

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': json.dumps(missing_clfrs_3),
            })
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data.decode())
            self.assertTrue(resp_data['message'].startswith(
                "The following labels are not registered with any"
                " classifiers:"))
            self.assertSetEqual(set(resp_data['missing_labels']),
                                set(missing_clfrs_3))

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': json.dumps(missing_clfrs_4),
            })
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data.decode())
            self.assertTrue(resp_data['message'].startswith(
                "The following labels are not registered with any"
                " classifiers:"))
            self.assertSetEqual(set(resp_data['missing_labels']),
                                set(missing_clfrs_4))

    def test_get_classifier_metadata_no_label(self):
        with self.app.test_client() as cli:
            #: :type: flask.wrappers.Response
            r = cli.get('/classifier_metadata')
            r_json = json.loads(r.data.decode())
            self.assertStatus(r, 400)
            self.assertMessage(r_json, "No label provided.")

    def test_get_classifier_metadata_invalid_label(self):
        with self.app.test_client() as cli:
            args = dict(label="no-valid-label")
            r = cli.get('/classifier_metadata', query_string=args)
            r_json = json.loads(r.data.decode())
            self.assertStatus(r, 404)
            self.assertMessage(r_json, "Label 'no-valid-label' does not refer "
                                       "to a classifier currently registered.")

    def test_get_classifier_labels_mocked(self):
        """
        Test that we can request the registered dummy classifiers class labels.
        Using mock objects to assert calls made.
        """
        expected_label = 'this-test-label'
        expected_class_labels = ['foo', 'bar', 'shazam']

        mock_classifier = mock.Mock(spec=Classifier)
        mock_classifier.get_labels = \
            mock.Mock(return_value=expected_class_labels)

        self.app.classifier_collection.labels = mock.Mock(
            return_value={expected_label}
        )
        self.app.classifier_collection.get_classifier = mock.Mock(
            return_value=mock_classifier
        )

        with self.app.test_client() as cli:
            args = dict(label=expected_label)
            r = cli.get('/classifier_metadata', query_string=args)

            self.app.classifier_collection.labels.assert_called_once_with()
            self.app.classifier_collection.get_classifier\
                    .assert_called_once_with(expected_label)
            mock_classifier.get_labels.assert_called_once_with()

            r_json = json.loads(r.data.decode())
            self.assertStatus(r, 200)
            self.assertMessage(r_json, "Success")
            self.assertIn('class_labels', r_json)
            self.assertListEqual(r_json['class_labels'], expected_class_labels)

    def test_get_classifier_labels(self):
        """
        Test that we can request the registered dummy classifiers class labels.
        """
        with self.app.test_client() as cli:
            args = dict(label=self.dummy_label)
            r = cli.get('/classifier_metadata', query_string=args)
            r_json = json.loads(r.data.decode())
            self.assertStatus(r, 200)
            self.assertMessage(r_json, "Success")
            self.assertIn('class_labels', r_json)
            self.assertSetEqual(set(r_json['class_labels']),
                                {'negative', 'positive'})

    def test_add_iqr_state_classifier_param_failures(self):
        test_bytes = b"some not used bytes"
        test_bytes_b64 = base64.b64encode(test_bytes).decode()
        test_label = "classifier-test-label"

        with self.app.test_client() as cli:
            # Missing Bytes
            rv = cli.post('/iqr_classifier')
            self.assertStatus(rv, 400)
            self.assertResponseMessageRegex(
                rv, "No state base64 data provided."
            )

            # Missing label for classifier
            rv = cli.post('/iqr_classifier', data={
                "bytes_b64": test_bytes_b64,
            })
            self.assertStatus(rv, 400)
            self.assertResponseMessageRegex(
                rv, "No descriptive label provided."
            )

            # Invalid lock flag value (not a boolean)
            rv = cli.post('/iqr_classifier', data={
                'bytes_b64': test_bytes_b64,
                'label': test_label,
                'lock_label': 'not-bool-convertible'
            })
            self.assertStatus(rv, 400)
            self.assertResponseMessageRegex(
                rv, "Invalid boolean value for 'lock_label'. Was given: "
            )

    def test_add_iqr_state_classifier_existing_label(self):
        test_label = 'duplicate-label'
        test_new_cfier_b64 = base64.b64encode(
            b"some not used bytes"
        )

        self.app.classifier_collection.add_classifier(
            test_label, DummySupervisedClassifier()
        )

        with self.app.test_client() as cli:
            rv = cli.post('/iqr_classifier', data={
                'bytes_b64': test_new_cfier_b64,
                'label': test_label,
            })
            self.assertStatus(rv, 400)
            self.assertResponseMessageRegex(
                rv, "Label already exists in classifier collection."
            )

    @mock.patch.dict(os.environ, {
        Pluggable.PLUGIN_ENV_VAR:
            OS_ENV_PATH_SEP.join([
                STUB_CLASSIFIER_MOD_PATH,
                'tests.web.classifier_service.dummy_descriptor_generator',
            ])
    })
    def test_add_iqr_state_classifier_simple(self):
        """
        Test calling IQR classifier add endpoint with a simple IQR Session
        serialization.
        """
        # Make a simple session with dummy adjudication descriptor elements
        rank_relevancy = mock.MagicMock(spec=RankRelevancy)
        iqrs = IqrSession(rank_relevancy, session_uid=str("0"))
        iqr_p1 = DescriptorMemoryElement('test', 0).set_vector([0])
        iqr_n1 = DescriptorMemoryElement('test', 1).set_vector([1])
        iqrs.adjudicate(
            new_positives=[iqr_p1], new_negatives=[iqr_n1]
        )

        test_iqrs_b64 = base64.b64encode(iqrs.get_state_bytes())
        test_label = 'test-label-08976azsdv'

        with mock.patch(STUB_CLASSIFIER_MOD_PATH +
                        ".DummySupervisedClassifier._train") as m_cfier_train:

            with self.app.test_client() as cli:
                rv = cli.post('/iqr_classifier', data={
                    'bytes_b64': test_iqrs_b64,
                    'label': test_label,
                })
                self.assertStatus(rv, 201)
                self.assertResponseMessageRegex(rv, "Finished training "
                                                    "IQR-session-based "
                                                    "classifier for label "
                                                    "'%s'." % test_label)

            m_cfier_train.assert_called_once_with(
                {'positive': {iqr_p1}, 'negative': {iqr_n1}}
            )
            # Collection should include initial dummy classifier and new iqr
            # classifier.
            self.assertEqual(len(self.app.classifier_collection.labels()), 2)
            self.assertIn(test_label, self.app.classifier_collection.labels())
