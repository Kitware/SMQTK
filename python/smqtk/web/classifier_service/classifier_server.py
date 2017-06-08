from __future__ import print_function

import base64

import flask
import six

from smqtk.algorithms import (
    get_classifier_impls,
    get_descriptor_generator_impls,
    SupervisedClassifier
)
from smqtk.algorithms.classifier import (
    ClassifierCollection,
)
from smqtk.iqr import IqrSession
from smqtk.representation import (
    ClassificationElementFactory,
    DescriptorElementFactory,
)
from smqtk.representation.data_element.memory_element import DataMemoryElement
import smqtk.utils.plugin
from smqtk.utils.web import make_response_json
import smqtk.web


class SmqtkClassifierService (smqtk.web.SmqtkWebApp):
    """
    Headless web-app providing a RESTful API for classifying new data against a
    set of statically and dynamically loaded classifier models.

    The focus of this service is an endpoint where the user can send the
    base64-encoded data (with content type) they wish to be classified and get
    back the classification results of all loaded classifiers applied to the
    description of that data. Data for classification sent to this service is
    expected to be in

    Saved IQR session state bytes/files may be POST'ed to an endpoint with a
    descriptive label to add to the suite of classifiers that are run for
    user-provided data. The supervised classifier implementation that is trained
    from this state is part of the server configuration.

    Configuration Notes
    -------------------
    * The configured classifiers must all handle the descriptors output by the
      descriptor generator algorithm. IQR states loaded into the server must
      come from a service that also used the same descriptor generation
      algorithm. Otherwise the classification of new data will not make sense
      given the configured models as well as exceptions may occur due to
      descriptor dimensionality issues.

    * The classifier configuration provided for input IQR states should not have
      model persistence parameters specified since these classifiers will be
      ephemeral. If persistence parameters *are* specified, then subsequent
      IQR-state-based classifier models will bash each other causing erroneously
      labeled duplicate results.

    """

    CONFIG_ENABLE_IQR_CLASSIFIER_REMOVAL = "enable_iqr_classifier_removal"
    CONFIG_CLASSIFIER_COLLECTION = "classifier_collection"
    CONFIG_CLASSIFICATION_FACTORY = "classification_factory"
    CONFIG_DESCRIPTOR_GENERATOR = "descriptor_generator"
    CONFIG_DESCRIPTOR_FACTORY = "descriptor_factory"
    CONFIG_IQR_CLASSIFIER = "iqr_state_classifier_config"

    DEFAULT_IQR_STATE_CLASSIFIER_KEY = '__default__'

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(SmqtkClassifierService, cls).get_default_config()

        c[cls.CONFIG_ENABLE_IQR_CLASSIFIER_REMOVAL] = False

        # Static classifier configurations
        c[cls.CONFIG_CLASSIFIER_COLLECTION] = \
            ClassifierCollection.get_default_config()
        # Classification element factory for new classification results.
        c[cls.CONFIG_CLASSIFICATION_FACTORY] = \
            ClassificationElementFactory.get_default_config()
        # Descriptor generator for new content
        c[cls.CONFIG_DESCRIPTOR_GENERATOR] = smqtk.utils.plugin.make_config(
            get_descriptor_generator_impls()
        )
        # Descriptor factory for new content descriptors
        c[cls.CONFIG_DESCRIPTOR_FACTORY] = \
            DescriptorElementFactory.get_default_config()
        # from-IQR-state *supervised* classifier configuration
        c[cls.CONFIG_IQR_CLASSIFIER] = smqtk.utils.plugin.make_config(
            get_classifier_impls(
                sub_interface=SupervisedClassifier
            )
        )

        return c

    def __init__(self, json_config):
        super(SmqtkClassifierService, self).__init__(json_config)

        self.enable_iqr_classifier_removal = \
            bool(json_config[self.CONFIG_ENABLE_IQR_CLASSIFIER_REMOVAL])

        # Convert configuration into SMQTK plugin instances.
        #   - Static classifier configurations.
        #       - Skip the example config key
        #   - Classification element factory
        #   - Descriptor generator
        #   - Descriptor element factory
        #   - from-IQR-state classifier configuration
        #       - There must at least be the default key defined for when no
        #         specific classifier type is specified at state POST.

        # Classifier collection + factor
        self.classification_factory = ClassificationElementFactory.from_config(
            json_config[self.CONFIG_CLASSIFICATION_FACTORY]
        )
        self.classifier_collection = ClassifierCollection.from_config(
            json_config[self.CONFIG_CLASSIFIER_COLLECTION]
        )

        # Descriptor generator + factory
        self.descriptor_factory = DescriptorElementFactory.from_config(
            json_config[self.CONFIG_DESCRIPTOR_FACTORY]
        )
        #: :type: smqtk.algorithms.DescriptorGenerator
        self.descriptor_gen = smqtk.utils.plugin.from_plugin_config(
            json_config[self.CONFIG_DESCRIPTOR_GENERATOR],
            smqtk.algorithms.get_descriptor_generator_impls()
        )

        # Classifier config for uploaded IQR states.
        self.iqr_state_classifier_config = \
            json_config[self.CONFIG_IQR_CLASSIFIER]
        self.iqr_classifier_collection = ClassifierCollection()

    def run(self, host=None, port=None, debug=False, **options):
        # REST API endpoint routes
        #
        # Example:
        # self.add_url_rule('/endpoint',
        #                   view_func=self.something,
        #                   methods=['GET'])
        #
        self.add_url_rule('/is_ready',
                          view_func=self.is_ready,
                          methods=['GET'])
        self.add_url_rule('/classifier_labels',
                          view_func=self.get_classifier_labels,
                          methods=['GET'])
        self.add_url_rule('/classify',
                          view_func=self.classify,
                          methods=['POST'])
        self.add_url_rule('/iqr_classifier',
                          view_func=self.get_iqr_classifier_labels,
                          methods=['GET'])
        self.add_url_rule('/iqr_classifier',
                          view_func=self.add_iqr_state_classifier,
                          methods=['POST'])
        if self.enable_iqr_classifier_removal:
            self.add_url_rule('/iqr_classifier',
                              view_func=self.del_iqr_state_classifier,
                              methods=['DELETE'])

        super(SmqtkClassifierService, self).run(host, port, debug, **options)

    # GET /is_ready
    # noinspection PyMethodMayBeStatic
    def is_ready(self):
        """
        Simple endpoint that just means this server is up and responding.
        """
        return make_response_json("Yes, I'm alive!")

    # GET /classifier_labels
    def get_classifier_labels(self):
        """
        Get the descriptive labels of the classifiers currently set to classify
        input data.

        Returns 200: {
            labels: list[str]
        }

        """
        # join labels from both collections.
        all_labels = (self.classifier_collection.labels() |
                      self.iqr_classifier_collection.labels())
        return make_response_json("All classifier labels.",
                                  labels=list(all_labels))

    # POST /classify
    def classify(self):
        """
        Describe and classify provided base64 data, returning results in JSON
        response.

        We expect the data to be transmitted in the body of the request in
        standard base64 encoding form ("bytes_b64" key). We look for the content
        type either as URL parameter or within the body ("content_type" key).

        Below is an example call to this endpoint via the ``requests`` python
        module, showing how base64 data is sent::

            import base64
            import requests
            data_bytes = "Load some content bytes here."
            requests.get('http://localhost:5000/classify',
                         data={'bytes_b64': base64.b64encode(data_bytes),
                               'content_type': 'text/plain'})

        With curl on the command line::

            $ curl -X POST localhost:5000/iqr_classifier -d label=some_label \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

        Curl may fail depending on the size of the file and how long your
        terminal allows argument lists.

        Data args:
            bytes_b64
                Bytes of the data to be described and classified in base64
                encoding.
            content_type
                The mimetype of the sent data.

        Possible error codes:
            400
                No bytes or label provided

        Returns: {
            ...
            result: {
                classifier-label: {
                    class-label: prob,
                    ...
                },
                ...
            }
        }

        """
        data_b64 = flask.request.form.get('bytes_b64', None)
        content_type = flask.request.form.get('content_type', None)

        if data_b64 is None:
            return make_response_json("No base-64 bytes provided.", 400)
        elif content_type is None:
            return make_response_json("No content type provided.", 400)

        data_bytes = base64.b64decode(data_b64.encode('utf-8'))
        self._log.debug("Length of byte data: %d" % len(data_bytes))

        data_elem = DataMemoryElement(data_bytes, content_type, readonly=True)
        descr_elem = self.descriptor_gen.compute_descriptor(
            data_elem, self.descriptor_factory
        )
        self._log.debug("Descriptor shape: %s", descr_elem.vector().shape)

        static_c_map = self.classifier_collection.classify(
            descr_elem, self.classification_factory
        )
        iqr_c_map = self.iqr_classifier_collection.classify(
            descr_elem, self.classification_factory
        )

        # Transform classification result into JSON
        c_json = {}
        for classifier_label, c_elem in six.iteritems(static_c_map):
            c_json[classifier_label] = c_elem.get_classification()
        for classifier_label, c_elem in six.iteritems(iqr_c_map):
            c_json[classifier_label] = c_elem.get_classification()

        return make_response_json('Finished classification.',
                                  result=c_json)

    # GET /iqr_classifier
    def get_iqr_classifier_labels(self):
        """
        Get the labels of the classifiers specifically added via uploaded
        IQR session states.

        Returns 200: {
            ...
            labels: list[str]
        }
        """
        return make_response_json(
            "IQR state-based classifier labels.",
            labels=list(self.iqr_classifier_collection.labels()),
        )

    # POST /iqr_classifier
    def add_iqr_state_classifier(self):
        """
        Train a classifier based on the user-provided IQR state file bytes in a
        base64 encoding, matched with a descriptive label of that classifier's
        topic.

        Since all IQR session classifiers end up only having two result classes
        (positive and negative), the topic of the classifier is encoded in the
        descriptive label the user applies to the classifier.

        Below is an example call to this endpoint via the ``requests`` python
        module, showing how base64 data is sent::

            import base64
            import requests
            data_bytes = "Load some content bytes here."
            requests.get('http://localhost:5000/classify',
                         data={'bytes_b64': base64.b64encode(data_bytes),
                               'content_type': 'text/plain'})

        With curl on the command line::

            $ curl -X POST localhost:5000/iqr_classifier -d label=some_label \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

        Curl may fail depending on the size of the file and how long your
        terminal allows argument lists.

        Form arguments:
            iqr_state_b64
                base64 encoding of the bytes of the IQR session state save file.
            label
                Descriptive label to apply to this classifier. This should not
                conflict with existing classifier labels.

        Returns 201.

        """
        data_b64 = (flask.request.form.get('bytes_b64', None) or
                    flask.request.args.get('bytes_b64', None))
        label = (flask.request.args.get('label', None) or
                 flask.request.form.get('label', None))

        if data_b64 is None or len(data_b64) == 0:
            return make_response_json("No state base64 data provided.", 400)
        elif label is None or len(label) == 0:
            return make_response_json("No descriptive label provided.", 400)

        # If the given label conflicts with one already in either
        # collection, fail.
        if label in self.classifier_collection.labels():
            return make_response_json("Label already exists in static "
                                      "classifier collection.", 400)
        elif label in self.iqr_classifier_collection.labels():
            return make_response_json("Label already exists in IQR "
                                      "classifier collection.", 400)

        # Create dummy IqrSession to extract pos/neg descriptors.
        iqrs = IqrSession()
        iqrs.set_state_bytes(base64.b64decode(data_b64.encode('utf-8')),
                             self.descriptor_factory)
        pos = iqrs.positive_descriptors | iqrs.external_positive_descriptors
        neg = iqrs.negative_descriptors | iqrs.external_negative_descriptors

        # Make a classifier instance from the stored config for IQR
        # session-based classifiers.
        #: :type: SupervisedClassifier
        classifier = smqtk.utils.plugin.from_plugin_config(
            self.iqr_state_classifier_config,
            get_classifier_impls(sub_interface=SupervisedClassifier)
        )
        classifier.train(positive=pos, negative=neg)

        try:
            self.iqr_classifier_collection.add_classifier(label, classifier)
        except ValueError:
            return make_response_json("Duplicate label ('%s') added during "
                                      "classifier training of provided IQR "
                                      "session state." % label, 400,
                                      label=label)

        return make_response_json("Finished training IQR-session-based "
                                  "classifier for label '%s'." % label,
                                  201,
                                  label=label)

    # DEL /iqr_classifier
    def del_iqr_state_classifier(self):
        """
        Remove an IQR state classifier by the given label.

        Form args:
            label
                Label of the IQR state classifier to remove.

        Possible error codes:
            400
                No IQR state classifier exists for the given label.

        Returns 200.

        """
        label = flask.request.form.get('label', None)
        if label is None or not label:
            return make_response_json("No label provided.", 400)
        elif label not in self.iqr_classifier_collection.labels():
            return make_response_json("Provided label does not refer to an "
                                      "IQR classifier currently registered.",
                                      404)

        self.iqr_classifier_collection.remove_classifier(label)

        return make_response_json("Removed IQR classifier with label '%s'."
                                  % label,
                                  removed_label=label)
