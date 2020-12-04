from __future__ import print_function

import base64
import binascii
import json
import pickle
import urllib.parse

import flask
import six

from smqtk.algorithms import (
    DescriptorGenerator,
    SupervisedClassifier
)
from smqtk.algorithms.classifier import (
    ClassifierCollection,
)
from smqtk.exceptions import MissingLabelError
from smqtk.iqr import IqrSession
from smqtk.representation import (
    ClassificationElementFactory,
    DescriptorElementFactory,
    DescriptorSet,
)
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.descriptor_set.memory import MemoryDescriptorSet
from smqtk.utils import probability
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)
from smqtk.utils.web import make_response_json
import smqtk.web


# Get expected JSON decode exception
# noinspection PyProtectedMember
if hasattr(flask.json._json, 'JSONDecodeError'):
    # noinspection PyProtectedMember
    JSON_DECODE_EXCEPTION = getattr(flask.json._json, 'JSONDecodeError')
else:
    # Exception thrown from ``json`` module.
    if six.PY2:
        JSON_DECODE_EXCEPTION = ValueError
    else:
        # noinspection PyUnresolvedReferences
        JSON_DECODE_EXCEPTION = json.JSONDecodeError


def labels_from_input(label_str):
    """
    Parse out a single or multiple labels from the provided JSON-encoded value.

    :param str label_str: JSON encoded value to parse from.

    :raises ValueError: Invalid label string.

    :return: List of string labels or None if no labels were provided or was
        not JSON parsable.
    :rtype: list[str]
    """
    if not label_str:
        return None
    try:
        labels = json.loads(label_str)
        if isinstance(labels, str):
            labels = [labels]
        elif isinstance(labels, list):
            for el in labels:
                if not isinstance(el, str):
                    raise ValueError("Label must be a list of strings or a "
                                     "single string: give a list of more "
                                     "than just strings (found type: "
                                     f"{type(el).__name__}).")
        else:
            raise ValueError("Label must be a list of strings or a single "
                             f"string (given type: {type(labels).__name__}).")
    except json.JSONDecodeError:
        # Unquoted strings aren't valid JSON. That is, a plain string
        # needs to be passed as '"label"' rather than just 'label' or
        # "label". However, we can be a bit more generous and just
        # allow such a string, but we have to place *some* restriction
        # on it. We use `urllib.quote` for this since essentially it
        # just checks to make sure that the string is made up of one
        # of the following types of characters:
        #
        #   - letters
        #   - numbers
        #   - spaces, underscores, periods, and dashes
        #
        # Since the concept of a "letter" is fraught with encoding and
        # locality issues, we simply let urllib make this decision for
        # us.

        # If label_str matches the url-encoded version of itself, go
        # ahead and use it
        if urllib.parse.quote(label_str, safe=' ') == label_str:
            labels = [label_str]
        else:
            raise ValueError("Label is not a properly formatted JSON nor a "
                             "simple string.")
    return labels


class SmqtkClassifierService (smqtk.web.SmqtkWebApp):
    """
    Headless web-app providing a RESTful API for classifying new data against
    a set of statically and dynamically loaded classifier models.

    The focus of this service is an endpoint where the user can send the
    base64-encoded data (with content type) they wish to be classified and get
    back the classification results of all loaded classifiers applied to the
    description of that data. Data for classification sent to this service is
    expected to be in

    Saved IQR session state bytes/files may be POST'ed to an endpoint with a
    descriptive label to add to the suite of classifiers that are run for
    user-provided data. The supervised classifier implementation that is
    trained from this state is part of the server configuration.

    Configuration Notes
    -------------------
    * The configured classifiers must all handle the descriptors output by the
      descriptor generator algorithm. IQR states loaded into the server must
      come from a service that also used the same descriptor generation
      algorithm. Otherwise the classification of new data will not make sense
      given the configured models as well as exceptions may occur due to
      descriptor dimensionality issues.

    * The classifier configuration provided for input IQR states should not
      have model persistence parameters specified since these classifiers will
      be ephemeral. If persistence parameters *are* specified, then subsequent
      IQR-state-based classifier models will bash each other causing
      erroneously labeled duplicate results.

    """

    CONFIG_ENABLE_CLASSIFIER_REMOVAL = "enable_classifier_removal"
    CONFIG_CLASSIFIER_COLLECTION = "classifier_collection"
    CONFIG_CLASSIFICATION_FACTORY = "classification_factory"
    CONFIG_DESCRIPTOR_GENERATOR = "descriptor_generator"
    CONFIG_DESCRIPTOR_FACTORY = "descriptor_factory"
    CONFIG_DESCRIPTOR_SET = "descriptor_set"
    CONFIG_IMMUTABLE_LABELS = "immutable_labels"
    CONFIG_IQR_CLASSIFIER = "iqr_state_classifier_config"

    DEFAULT_IQR_STATE_CLASSIFIER_KEY = '__default__'

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(SmqtkClassifierService, cls).get_default_config()

        c[cls.CONFIG_ENABLE_CLASSIFIER_REMOVAL] = False

        # Static classifier configurations
        c[cls.CONFIG_CLASSIFIER_COLLECTION] = \
            ClassifierCollection.get_default_config()
        # Classification element factory for new classification results.
        c[cls.CONFIG_CLASSIFICATION_FACTORY] = \
            ClassificationElementFactory.get_default_config()
        # Descriptor generator for new content
        c[cls.CONFIG_DESCRIPTOR_GENERATOR] = make_default_config(
            DescriptorGenerator.get_impls()
        )
        # Descriptor factory for new content descriptors
        c[cls.CONFIG_DESCRIPTOR_FACTORY] = \
            DescriptorElementFactory.get_default_config()
        # Optional Descriptor set for "included" descriptors referenceable by
        # UID.
        c[cls.CONFIG_DESCRIPTOR_SET] = make_default_config(
            DescriptorSet.get_impls()
        )
        # from-IQR-state *supervised* classifier configuration
        c[cls.CONFIG_IQR_CLASSIFIER] = make_default_config(
            SupervisedClassifier.get_impls()
        )
        c[cls.CONFIG_IMMUTABLE_LABELS] = []

        return c

    def __init__(self, json_config):
        super(SmqtkClassifierService, self).__init__(json_config)

        self.enable_classifier_removal = \
            bool(json_config[self.CONFIG_ENABLE_CLASSIFIER_REMOVAL])

        self.immutable_labels = set(json_config[self.CONFIG_IMMUTABLE_LABELS])

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
        self.classification_factory = \
            ClassificationElementFactory.from_config(
                json_config[self.CONFIG_CLASSIFICATION_FACTORY]
            )
        #: :type: ClassifierCollection
        self.classifier_collection = ClassifierCollection.from_config(
            json_config[self.CONFIG_CLASSIFIER_COLLECTION]
        )

        # Descriptor generator + factory
        self.descriptor_factory = DescriptorElementFactory.from_config(
            json_config[self.CONFIG_DESCRIPTOR_FACTORY]
        )
        #: :type: smqtk.algorithms.DescriptorGenerator
        self.descriptor_gen = from_config_dict(
            json_config[self.CONFIG_DESCRIPTOR_GENERATOR],
            smqtk.algorithms.DescriptorGenerator.get_impls()
        )

        # Descriptor set bundled for classification-by-UID.
        try:
            self.descriptor_set = from_config_dict(
                json_config.get(self.CONFIG_DESCRIPTOR_SET, {}),
                DescriptorSet.get_impls()
            )
        except ValueError:
            # Default empty set.
            self.descriptor_set = MemoryDescriptorSet()

        # Classifier config for uploaded IQR states.
        self.iqr_state_classifier_config = \
            json_config[self.CONFIG_IQR_CLASSIFIER]

        self.add_routes()

    def add_routes(self):
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
        self.add_url_rule('/classifier_metadata',
                          view_func=self.get_classifier_metadata,
                          methods=['GET'])
        self.add_url_rule('/classify_uids',
                          view_func=self.classify_uids,
                          methods=['POST'])
        self.add_url_rule('/classify',
                          view_func=self.classify,
                          methods=['POST'])
        self.add_url_rule('/classifier',
                          view_func=self.get_classifier,
                          methods=['GET'])
        self.add_url_rule('/classifier',
                          view_func=self.add_classifier,
                          methods=['POST'])
        self.add_url_rule('/iqr_classifier',
                          view_func=self.add_iqr_state_classifier,
                          methods=['POST'])
        if self.enable_classifier_removal:
            self.add_url_rule('/classifier',
                              view_func=self.del_classifier,
                              methods=['DELETE'])

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
        Get the descriptive labels of the classifiers currently set to
        classify input data.

        Returns 200: {
            labels: list[str]
        }

        """
        all_labels = self.classifier_collection.labels()
        return make_response_json("Classifier labels.",
                                  labels=list(all_labels))

    # GET /classifier_metadata
    def get_classifier_metadata(self):
        """
        Get metadata associated with a specific classifier instance referred to
        by label.

        URL Arguments:
            label
                Reference label for a specific classifier to query.

        Returns code 200 on success and the JSON return object: {
            ...
            // Sequence of class labels that this classifier can classify
            // descriptors into.  This includes the negative label.
            class_labels=<list[str]>
        }

        """
        label = flask.request.values.get('label', default=None)
        if label is None or not label:
            return make_response_json("No label provided.", return_code=400)
        elif label not in self.classifier_collection.labels():
            return make_response_json("Label '%s' does not refer to a "
                                      "classifier currently registered."
                                      % label,
                                      return_code=404,
                                      label=label)
        class_labels = \
            self.classifier_collection.get_classifier(label).get_labels()
        return make_response_json("Success", return_code=200,
                                  class_labels=class_labels)

    # POST /classify_uids
    def classify_uids(self):
        """
        Given a list of descriptor UIDs, we attempt to retrieve descriptor
        vectors from our configured descriptor index and classify those vectors
        against all currently stored classifiers (optionally a list of
        requested classifiers), returning a map of classifier descriptive
        labels to their class-to-probability results.

        TODO: Add `adjustment` param from POST /classify

        Arguments
            uid_list:
                JSON list of UIDs of descriptors in the configured descriptor
                set to classify. These should be strings or integers depending
                on the keys used in the configured descriptor set (usually
                strings).
            label:
                Optional string label or JSON list of string labels defining
                specific classifiers to use for inferencing against the
                descriptors.

        Possible error codes:
            400
                No UIDs provided, or provided labels are malformed.
            404
                Label or labels provided do not match any registered
                classifier

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
        if self.classifier_collection.size() == 0:
            return make_response_json("No classifiers currently loaded.", 200,
                                      result={})

        uid_list = flask.request.values.get('uid_list')
        if uid_list is None:
            return make_response_json("No UIDs provided.", 400)
        try:
            uid_list = json.loads(uid_list)
        except json.JSONDecodeError:
            return make_response_json("Failed to parse JSON list of UIDs.",
                                      400)
        if not uid_list:
            return make_response_json("No UIDs provided.", 400)

        try:
            # We could technically pass `labels_from_input` as the `type=`
            # value in `.get()` but the method eats ValueErrors raised.
            labels = labels_from_input(
                flask.request.values.get('label', default=None))
        except ValueError as ex:
            return make_response_json(f"Invalid label(s) specified: {ex}", 400)

        # Label list has been parsed at this point. Make sure its contents
        # meshes with available classifiers before retrieving descriptors.
        if labels is not None:
            missing_labels = (set(labels) -
                              set(self.classifier_collection.labels()))
            if missing_labels:
                return make_response_json(
                    "The following labels are not registered with any "
                    "classifiers: " + ", ".join(map(repr, missing_labels)),
                    404,
                    missing_labels=list(missing_labels))

        try:
            vec_list = self.descriptor_set.get_many_vectors(uid_list)
        except KeyError:
            return make_response_json("One or more input UIDs did not exist in"
                                      " the configured descriptor set!", 400)
        pred_map = self.classifier_collection.classify_arrays(
            vec_list, labels=labels
        )

        # TODO: Add `adjustment` functionality like from `POST /classify`.

        return make_response_json("", 200, result=pred_map)

    # POST /classify
    def classify(self):
        """
        Given a file's bytes (standard base64-format) and content mimetype,
        describe and classify the content against all currently stored
        classifiers (optionally a list of requested classifiers), returning a
        map of classifier descriptive labels to their class-to-probability
        results.

        We expect the data to be transmitted in the body of the request in
        standard base64 encoding form ("bytes_b64" key). We look for the
        content type either as URL parameter or within the body
        ("content_type" key).

        Below is an example call to this endpoint via the ``requests`` python
        module, showing how base64 data is sent::

            import base64
            import requests
            data_bytes = "Load some content bytes here."
            requests.post('http://localhost:5000/classify',
                          data={'bytes_b64': base64.b64encode(data_bytes),
                                'content_type': 'text/plain'})

        With curl on the command line::

            $ curl -X POST localhost:5000/classify \
                -d "content_type=text/plain" \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

            # If this fails, you may wish to encode the file separately and
            # use the file reference syntax instead:

            $ base64 -w0 /path/to/file > /path/to/file.b64
            $ curl -X POST localhost:5000/classify \
                -d "content_type=text/plain" \
                --data-urlencode bytes_64@/path/to/file.b64

        Optionally, the `label` parameter can be provided to limit the results
        of classification to a set of classifiers::

            $ curl -X POST localhost:5000/classify \
                -d "content_type=text/plain" \
                -d 'label=["some_label","other_label"]' \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

            # If this fails, you may wish to encode the file separately and
            # use the file reference syntax instead:

            $ base64 -w0 /path/to/file > /path/to/file.b64
            $ curl -X POST localhost:5000/classify \
                -d "content_type=text/plain" \
                -d 'label=["some_label","other_label"]' \
                --data-urlencode bytes_64@/path/to/file.b64

        Data/Form arguments:
            bytes_b64
                Bytes in the standard base64 encoding to be described and
                classified.
            content_type
                The mimetype of the sent data.
            label
                (Optional) JSON-encoded label or list of labels
            adjustment
                (Optional) JSON-encoded dictionary of labels to floats. Higher
                values lower the gain on the class and therefore correspond to
                higher precision (and lower recall) for the class (and higher
                recall/lower precision for other classes). This translates
                to calling ``smqtk.utils.probability.adjust_proba``.

        Possible error codes:
            400
                No bytes provided, or provided labels are malformed
            404
                Label or labels provided do not match any registered
                classifier

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
        data_b64 = flask.request.values.get('bytes_b64', default=None)
        content_type = flask.request.values.get('content_type', default=None)
        label_str = flask.request.values.get('label', default=None)
        adjustment_str = flask.request.values.get('adjustment', default=None)

        try:
            labels = labels_from_input(label_str)
        except ValueError as ex:
            return make_response_json(f"Invalid label(s) specified: {ex}", 400)

        # Collect optional result probability adjustment values
        #: :type: dict[collections.abc.Hashable, float]
        adjustments = {}
        if adjustment_str is not None:
            try:
                #: :type: dict[collections.abc.Hashable, float]
                adjustments = flask.json.loads(adjustment_str)

                for label, val in six.iteritems(adjustments):
                    if not isinstance(label, six.string_types):
                        return make_response_json(
                            "Adjustment label '%s' is not a string type."
                            % label,
                            400)
                    if not isinstance(val, (int, float)):
                        return make_response_json(
                            "Adjustment value %s for label '%s' is not an int "
                            "or float" % (val, label),
                            400)
            except JSON_DECODE_EXCEPTION:
                return make_response_json(
                    "Adjustment(s) are not properly formatted JSON.", 400)

        if data_b64 is None:
            return make_response_json("No base-64 bytes provided.", 400)
        elif content_type is None:
            return make_response_json("No content type provided.", 400)

        data_bytes = base64.b64decode(data_b64.encode('utf-8'))
        self._log.debug("Length of byte data: %d" % len(data_bytes))

        data_elem = DataMemoryElement(data_bytes, content_type, readonly=True)
        descr_elem = self.descriptor_gen.generate_one_element(
            data_elem, descr_factory=self.descriptor_factory
        )
        self._log.debug("Descriptor shape: %s", descr_elem.vector().shape)

        try:
            clfr_map = self.classifier_collection.classify(
                descr_elem, labels=labels,
                factory=self.classification_factory)
        except MissingLabelError as ex:
            return make_response_json(
                "The following labels are not registered with any"
                " classifiers: '%s'"
                % "', '".join(ex.labels),
                404,
                missing_labels=list(ex.labels))

        # Transform classification result into JSON
        c_json = {}
        for classifier_label, c_elem in six.iteritems(clfr_map):
            prediction = c_elem.get_classification()
            if adjustments:
                proba_labels = list(prediction.keys())
                proba = [prediction[k] for k in proba_labels]
                # Use opposite of adjustments, because we already set the
                # convention of "higher: precision, lower: recall"
                adj = [-adjustments.get(label, 0.0) for label in proba_labels]
                adj_proba = probability.adjust_proba(proba, adj)
                prediction = dict(zip(proba_labels, adj_proba[0]))
            c_json[classifier_label] = prediction

        return make_response_json('Finished classification.',
                                  result=c_json)

    # GET /classifier
    def get_classifier(self):
        """
        Download the classifier corresponding to the provided label, pickled
        and encoded in standard base64 encoding.

        Below is an example call to this endpoint via the ``requests`` python
        module::

            import base64
            import requests
            from six.moves import cPickle as pickle

            r = requests.get('http://localhost:5000/classifier',
                             data={'label': 'some_label'})
            data_bytes = base64.b64decode(r.content)
            classifier = pickle.loads(data_bytes)

        With curl on the command line::

            $ curl -X GET localhost:5000/classifier -d label=some_label | \
                base64 -d > /path/to/file.pkl

        Data args:
            label
                Label of the requested classifier

        Possible error codes:
            400
                No label provided
            404
                Label does not refer to a registered classifier

        Returns: The pickled and encoded classifier
        """
        label = flask.request.values.get('label', default=None)
        if label is None or not label:
            return make_response_json("No label provided.", 400)
        elif label not in self.classifier_collection.labels():
            return make_response_json("Label '%s' does not refer to a "
                                      "classifier currently registered."
                                      % label,
                                      404,
                                      label=label)

        clfr = self.classifier_collection.get_classifier(label)

        try:
            return base64.b64encode(pickle.dumps(clfr)), 200
        except pickle.PicklingError:
            return make_response_json("Classifier corresponding to label "
                                      "'%s' cannot be pickled." % label,
                                      500,
                                      label=label)

    # POST /iqr_classifier
    def add_iqr_state_classifier(self):
        """
        Train a classifier based on the user-provided IQR state file bytes in
        a base64 encoding, matched with a descriptive label of that
        classifier's topic.

        Since all IQR session classifiers end up only having two result
        classes (positive and negative), the topic of the classifier is
        encoded in the descriptive label the user applies to the classifier.

        Below is an example call to this endpoint via the ``requests`` python
        module, showing how base64 data is sent::

            import base64
            import requests
            data_bytes = "Load some content bytes here."
            requests.get('http://localhost:5000/iqr_classifier',
                         data={'bytes_b64': base64.b64encode(data_bytes),
                               'label': 'some_label'})

        With curl on the command line::

            $ curl -X POST localhost:5000/iqr_classifier \
                -d "label=some_label" \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

            # If this fails, you may wish to encode the file separately and
            # use the file reference syntax instead:

            $ base64 -w0 /path/to/file > /path/to/file.b64
            $ curl -X POST localhost:5000/iqr_classifier -d label=some_label \
                --data-urlencode bytes_64@/path/to/file.b64

        To lock this classifier and guard it against deletion, add
        "lock_label=true"::

            $ curl -X POST localhost:5000/iqr_classifier \
                -d "label=some_label" \
                -d "lock_label=true" \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

        Form arguments:
            iqr_state_b64
                base64 encoding of the bytes of the IQR session state save
                file.
            label
                Descriptive label to apply to this classifier. This should not
                conflict with existing classifier labels.
            lock_label
                If 'true', disallow deletion of this label. If 'false', allow
                deletion of this label. Only has an effect if deletion is
                enabled for this service. (Default: 'false')

        Returns 201.

        """
        data_b64 = flask.request.values.get('bytes_b64', default=None)
        label = flask.request.values.get('label', default=None)
        lock_clfr_str = flask.request.values.get('lock_label',
                                                 default='false')

        if data_b64 is None or len(data_b64) == 0:
            return make_response_json("No state base64 data provided.", 400)
        elif label is None or len(label) == 0:
            return make_response_json("No descriptive label provided.", 400)
        try:
            lock_clfr = bool(flask.json.loads(lock_clfr_str))
        except JSON_DECODE_EXCEPTION:
            return make_response_json("Invalid boolean value for"
                                      " 'lock_label'. Was given: '%s'"
                                      % lock_clfr_str,
                                      400)
        try:
            # Using urlsafe version because it handles both regular and urlsafe
            # alphabets.
            data_bytes = base64.urlsafe_b64decode(data_b64.encode('utf-8'))
        except (TypeError, binascii.Error) as ex:
            return make_response_json("Invalid base64 input: %s" % str(ex)), \
                   400

        # If the given label conflicts with one already in the collection,
        # fail.
        if label in self.classifier_collection.labels():
            return make_response_json(
                "Label already exists in classifier collection.", 400)

        # Create dummy IqrSession to extract pos/neg descriptors.
        iqrs = IqrSession(None)
        iqrs.set_state_bytes(data_bytes, self.descriptor_factory)
        pos = iqrs.positive_descriptors | iqrs.external_positive_descriptors
        neg = iqrs.negative_descriptors | iqrs.external_negative_descriptors
        del iqrs

        # Make a classifier instance from the stored config for IQR
        # session-based classifiers.
        # noinspection PyTypeChecker
        #: :type: SupervisedClassifier
        classifier = from_config_dict(
            self.iqr_state_classifier_config,
            SupervisedClassifier.get_impls()
        )
        classifier.train(class_examples={'positive': pos, 'negative': neg})

        try:
            self.classifier_collection.add_classifier(label, classifier)

            # If we're allowing deletions, get the lock flag from the form and
            # set it for this classifier
            if self.enable_classifier_removal and lock_clfr:
                self.immutable_labels.add(label)

        except ValueError as e:
            if e.args[0].find('JSON') > -1:
                return make_response_json("Tried to parse malformed JSON in "
                                          "form argument.", 400)
            return make_response_json("Duplicate label ('%s') added during "
                                      "classifier training of provided IQR "
                                      "session state." % label, 400,
                                      label=label)

        return make_response_json("Finished training IQR-session-based "
                                  "classifier for label '%s'." % label,
                                  201,
                                  label=label)

    # POST /classifier
    def add_classifier(self):
        """
        Upload a **trained** classifier pickled and encoded in standard base64
        encoding, matched with a descriptive label of that classifier's topic.

        Since all classifiers have only two result classes (positive and
        negative), the topic of the classifier is encoded in the descriptive
        label the user applies to the classifier.

        Below is an example call to this endpoint via the ``requests`` python
        module, showing how base64 data is sent::

            import base64
            import requests
            data_bytes = "Load some content bytes here."
            requests.post('http://localhost:5000/classifier',
                          data={'bytes_b64': base64.b64encode(data_bytes),
                                'label': 'some_label'})

        With curl on the command line::

            $ curl -X POST localhost:5000/classifier -d label=some_label \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

            # If this fails, you may wish to encode the file separately and
            # use the file reference syntax instead:

            $ base64 -w0 /path/to/file.pkl > /path/to/file.pkl.b64
            $ curl -X POST localhost:5000/classifier -d label=some_label \
                --data-urlencode bytes_64@/path/to/file.pkl.b64

        To lock this classifier and guard it against deletion, add
        "lock_label=true"::

            $ curl -X POST localhost:5000/classifier \
                -d "label=some_label" \
                -d "lock_label=true" \
                --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file.pkl)"

        Data/Form arguments:
            bytes_b64
                Bytes, in the standard base64 encoding, of the pickled
                classifier.
            label
                Descriptive label to apply to this classifier. This should not
                conflict with existing classifier labels.
            lock_label
                If 'true', disallow deletion of this label. If 'false', allow
                deletion of this label. Only has an effect if deletion is
                enabled for this service. (Default: 'false')

        Possible error codes:
            400
                May mean one of:
                    - No pickled classifier base64 data or label provided.
                    - Label provided is in conflict with an existing label in
                    the classifier collection.

        Returns code 201 on success and the message: {
            label: <str>
        }

        """
        clfr_b64 = flask.request.values.get('bytes_b64', default=None)
        label = flask.request.values.get('label', default=None)
        lock_clfr_str = flask.request.values.get('lock_label',
                                                 default='false')

        if clfr_b64 is None or len(clfr_b64) == 0:
            return make_response_json("No state base64 data provided.", 400)
        elif label is None or len(label) == 0:
            return make_response_json("No descriptive label provided.", 400)
        try:
            # This can throw a ValueError if lock_clfr is malformed JSON
            lock_clfr = bool(flask.json.loads(lock_clfr_str))
        except JSON_DECODE_EXCEPTION:
            return make_response_json("Invalid boolean value for"
                                      " 'lock_label'. Was given: '%s'"
                                      % lock_clfr_str,
                                      400)

        # If the given label conflicts with one already in the collection,
        # fail.
        if label in self.classifier_collection.labels():
            return make_response_json("Label '%s' already exists in"
                                      " classifier collection." % label,
                                      400,
                                      label=label)

        clfr = pickle.loads(base64.b64decode(clfr_b64.encode('utf-8')))

        try:
            self.classifier_collection.add_classifier(label, clfr)

            # If we're allowing deletions, get the lock flag from the form
            # and set it for this classifier
            if self.enable_classifier_removal and lock_clfr:
                self.immutable_labels.add(label)

        except ValueError:
            return make_response_json("Data added for label '%s' is not a"
                                      " Classifier." % label,
                                      400,
                                      label=label)

        return make_response_json("Uploaded classifier for label '%s'."
                                  % label,
                                  201,
                                  label=label)

    # DEL /classifier
    def del_classifier(self):
        """
        Remove a classifier by the given label.

        Form args:
            label
                Label of the classifier to remove.

        Possible error codes:
            400
                No classifier exists for the given label.

        Returns 200.

        """
        label = flask.request.values.get('label', default=None)
        if label is None or not label:
            return make_response_json("No label provided.", 400)
        elif label not in self.classifier_collection.labels():
            return make_response_json("Label '%s' does not refer to a"
                                      " classifier currently registered."
                                      % label,
                                      404,
                                      label=label)
        elif label in self.immutable_labels:
            return make_response_json("Label '%s' refers to a classifier"
                                      " that is immutable." % label,
                                      405,
                                      label=label)

        self.classifier_collection.remove_classifier(label)

        return make_response_json("Removed classifier with label '%s'."
                                  % label,
                                  removed_label=label)
