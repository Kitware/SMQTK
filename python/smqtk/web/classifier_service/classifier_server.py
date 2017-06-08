from __future__ import print_function

import base64

import flask
import six

from smqtk.algorithms import (
    get_classifier_impls,
    get_descriptor_generator_impls,
)
from smqtk.algorithms.classifier import (
    ClassifierCollection,
)
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
                sub_interface=smqtk.algorithms.SupervisedClassifier
            )
        )

        return c

    def __init__(self, json_config):
        super(SmqtkClassifierService, self).__init__(json_config)

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
                          methods=['GET'])

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
        return make_response_json("Success",
                                  labels=list(all_labels))

    # GET /classify
    def classify(self):
        """
        Describe and classify provided base64 data.

        We expect the data to be transmitted in the body of the request in
        standard base64 encoding form ("bytes_b64" key). We look for the content
        type either as URL parameter or within the body ("content_type" key).

        Below is an example call to this endpoint via the ``requests`` python
        module::

            import base64
            import requests
            data_bytes = "Load some content bytes here."
            requests.get('http://localhost:5000/classify',
                         data={'bytes_b64': base64.b64encode(data_bytes),
                               'content_type': 'text/plain'})

        Data args:
            bytes_b64
                Bytes of the data to be described and classified in base64
                encoding.
            content_type
                The mimetype of the sent data.

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
        data_b64 = flask.request.form.get('bytes_b64', None) or \
                   flask.request.args.get('bytes_b64', None)
        content_type = flask.request.args.get('content_type', None) or \
                       flask.request.form.get('content_type', None)

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

        c_map = self.classifier_collection.classify(descr_elem,
                                                    self.classification_factory)

        # Transform classification result into JSON
        c_json = {}
        for classifier_label, c_elem in six.iteritems(c_map):
            c_json[classifier_label] = c_elem.get_classification()

        return make_response_json('Finished classification.',
                                  result=c_json)
