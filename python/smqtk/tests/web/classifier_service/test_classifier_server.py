import base64
import json
import mock
import os
import unittest

from six.moves import cPickle as pickle

from .dummy_classifier import DummyClassifier
from smqtk.web.classifier_service.classifier_server import \
    SmqtkClassifierService


class TestClassifierService (unittest.TestCase):

    # noinspection PyUnresolvedReferences
    @mock.patch.dict(os.environ, {
        'CLASSIFIER_PATH':
            'smqtk.tests.web.classifier_service.dummy_classifier',
        'DESCRIPTOR_GENERATOR_PATH':
            'smqtk.tests.web.classifier_service.dummy_descriptor_generator'
    })
    def setUp(self):
        super(TestClassifierService, self).setUp()
        self.config = SmqtkClassifierService.get_default_config()

        self.config['classification_factory']['type'] = \
            'MemoryClassificationElement'
        del self.config['classification_factory']['FileClassificationElement']

        del self.config['classifier_collection']['__example_label__']
        self.dummy_label = 'dummy'
        self.config['classifier_collection'][self.dummy_label] = {
            'DummyClassifier': {},
            'type': 'DummyClassifier'
        }
        self.config['immutable_labels'] = [self.dummy_label]

        self.config['descriptor_factory']['type'] = 'DescriptorMemoryElement'
        del self.config['descriptor_factory']['DescriptorFileElement']

        self.config['descriptor_generator'] = {
            'DummyDescriptorGenerator': {},
            'type': 'DummyDescriptorGenerator'
        }

        self.config['iqr_state_classifier_config']['type'] = \
            'LibSvmClassifier'

        self.config['enable_classifier_removal'] = True

        self.config['flask_app'] = {}
        del self.config['server']

        self.app = SmqtkClassifierService(json_config=self.config)
        
    def assertStatus(self, rv, code):
        self.assertEqual(rv.status_code, code)
        
    def assertMessage(self, resp_data, message):
        self.assertEqual(resp_data['message'], message)

    def test_server_alive(self):
        rv = self.app.test_client().get('/is_ready')

        self.assertStatus(rv, 200)
        resp_data = json.loads(rv.data)
        self.assertMessage(resp_data, "Yes, I'm alive!")

    def test_preconfigured_labels(self):
        rv = self.app.test_client().get('/classifier_labels')

        self.assertStatus(rv, 200)
        resp_data = json.loads(rv.data)
        self.assertMessage(resp_data, "Classifier labels.")
        self.assertListEqual(resp_data['labels'], ['dummy'])

    def test_one_classify(self):
        results_exp = dict(positive=0.5, negative=0.5)
        label = 'dummy'
        content_type = 'text/plain'
        element = base64.b64encode('TEST ELEMENT')

        with self.app.test_client() as cli:
            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
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

    def test_multiple_classify(self):
        content_type = 'text/plain'
        element = base64.b64encode('TEST ELEMENT')
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
            resp_data = json.loads(rv.data)
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
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data, "Finished classification.")
            self.assertDictEqual(resp_data['result'][new_label], results_exp)

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': element,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data, "Finished classification.")
            self.assertDictEqual(resp_data['result'][old_label], results_exp)
            self.assertDictEqual(resp_data['result'][new_label], results_exp)

    def test_get_add_del_classifier(self):
        old_label = 'dummy'
        new_label = 'dummy2'

        with self.app.test_client() as cli:
            rv = cli.get('/classifier', data={
                'label': old_label,
            })
            self.assertStatus(rv, 200)
            enc_data = rv.data

            rv = cli.post('/classifier', data={
                'label': new_label,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 201)
            resp_data = json.loads(rv.data)
            self.assertEqual(resp_data["message"],
                             "Uploaded classifier for label '%s'."
                             % new_label)
            self.assertEqual(resp_data["label"], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']),
                                {old_label, new_label})

            rv = cli.delete('/classifier', data={
                'label': new_label,
            })
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
            self.assertEqual(resp_data["message"],
                             "Removed classifier with label '%s'."
                             % new_label)
            self.assertEqual(resp_data["removed_label"], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']), {old_label})

    def test_add_imm_del_classifier(self):
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
            resp_data = json.loads(rv.data)
            self.assertEqual(resp_data["message"],
                             "Uploaded classifier for label '%s'."
                             % new_label)
            self.assertEqual(resp_data["label"], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data, "Classifier labels.")
            self.assertSetEqual(set(resp_data['labels']),
                                {old_label, new_label})

            rv = cli.delete('/classifier', data={
                'label': new_label,
            })
            self.assertStatus(rv, 405)
            resp_data = json.loads(rv.data)
            self.assertEqual(resp_data["message"],
                             "Label '%s' refers to a classifier that is"
                             " immutable." % new_label)
            self.assertEqual(resp_data['label'], new_label)

            rv = cli.get('/classifier_labels')
            self.assertStatus(rv, 200)
            resp_data = json.loads(rv.data)
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
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data,
                               "Label '%s' already exists in classifier"
                               " collection." % old_label)
            self.assertEqual(resp_data['label'], old_label)

            rv = cli.post('/classifier', data={'label': old_label})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "No state base64 data provided.")

            rv = cli.post('/classifier', data={'bytes_b64': enc_data})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "No descriptive label provided.")

            rv = cli.post('/classifier', data={
                'label': old_label,
                'lock_label': lock_clfr_str,
                'bytes_b64': enc_data,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "Invalid boolean value for 'lock_label'."
                               " Was given: '%s'" % lock_clfr_str)

            rv = cli.post('/classifier', data={
                'label': new_label,
                'bytes_b64': bad_data,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "Data added for label '%s' is not a"
                               " Classifier." % new_label)

    def test_del_classifier_failures(self):
        old_label = 'dummy'
        new_label = 'dummy2'

        with self.app.test_client() as cli:
            rv = cli.delete('/classifier', data={})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data), "No label provided.")

            rv = cli.delete('/classifier', data={'label': old_label})
            self.assertStatus(rv, 405)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data,
                               "Label '%s' refers to a classifier that is"
                               " immutable." % old_label)
            self.assertEqual(resp_data['label'], old_label)

            rv = cli.delete('/classifier', data={'label': new_label})
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data,
                               "Label '%s' does not refer to a classifier"
                               " currently registered." % new_label)
            self.assertEqual(resp_data['label'], new_label)

    def test_get_classifier_failures(self):
        label = 'dummy2'

        with self.app.test_client() as cli:
            rv = cli.get('/classifier', data={})
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data), "No label provided.")

            rv = cli.get('/classifier', data={'label': label})
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data)
            self.assertMessage(resp_data,
                               "Label '%s' does not refer to a classifier"
                               " currently registered." % label)
            self.assertEqual(resp_data['label'], label)

    def test_classify_failures(self):
        content_type = 'text/plain'
        bytes_b64 = base64.b64encode('TEST ELEMENT')
        label_invalid_json_failure = '['
        label_valid_json_failure = '{}'
        label_valid_json_list_failure = '["test", {}]'
        missing_clfrs_1 = ['dummy', 'foo']
        missing_clfrs_2 = ['dummy', 'foo', 'bar']
        missing_clfrs_3 = ['foo']
        missing_clfrs_4 = ['foo', 'bar']

        with self.app.test_client() as cli:
            rv = cli.post('/classify', data={
                'content_type': content_type,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "No base-64 bytes provided.")

            rv = cli.post('/classify', data={
                'bytes_b64': bytes_b64,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "No content type provided.")

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': label_invalid_json_failure,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "Label(s) are not properly formatted JSON.")

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': label_valid_json_failure,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "Label must be a list of strings or a single"
                               " string.")

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': label_valid_json_list_failure,
            })
            self.assertStatus(rv, 400)
            self.assertMessage(json.loads(rv.data),
                               "Label must be a list of strings or a single"
                               " string.")

            rv = cli.post('/classify', data={
                'content_type': content_type,
                'bytes_b64': bytes_b64,
                'label': json.dumps(missing_clfrs_1),
            })
            self.assertStatus(rv, 404)
            resp_data = json.loads(rv.data)
            self.assert_(resp_data['message'].startswith(
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
            resp_data = json.loads(rv.data)
            self.assert_(resp_data['message'].startswith(
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
            resp_data = json.loads(rv.data)
            self.assert_(resp_data['message'].startswith(
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
            resp_data = json.loads(rv.data)
            self.assert_(resp_data['message'].startswith(
                "The following labels are not registered with any"
                " classifiers:"))
            self.assertSetEqual(set(resp_data['missing_labels']),
                                set(missing_clfrs_4))
