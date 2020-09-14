import itertools
import threading
from typing import Iterable, Sequence

import numpy as np

from smqtk.exceptions import MissingLabelError
from smqtk.utils import SmqtkObject
from smqtk.utils.configuration \
    import Configurable, make_default_config, from_config_dict, to_config_dict
from smqtk.utils.dict import merge_dict

from ._defaults import DFLT_CLASSIFIER_FACTORY
from ._interface_classifier import Classifier


class ClassifierCollection (SmqtkObject, Configurable):
    """
    A collection of descriptively-labeled classifier instances for the purpose
    of applying all stored classifiers to one or more input descriptor
    elements.

    TODO: [optionally?] map a classification element factory per classifier.

    """

    EXAMPLE_KEY = '__example_label__'

    @classmethod
    def get_default_config(cls):
        c = super(ClassifierCollection, cls).get_default_config()

        # We list the label-classifier mapping on one level, so remove the
        # nested map parameter that can optionally be used in the constructor.
        del c['classifiers']

        # Add slot of a list of classifier plugin specifications
        c[cls.EXAMPLE_KEY] = make_default_config(Classifier.get_impls())

        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless an instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: ClassifierCollection

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        classifier_map = {}

        # Copying list of keys so we can update the dictionary as we loop.
        for label in list(config_dict.keys()):
            # Skip the example section.
            if label == cls.EXAMPLE_KEY:
                continue

            classifier_config = config_dict[label]
            classifier = from_config_dict(classifier_config,
                                          Classifier.get_impls())
            classifier_map[label] = classifier

        # Don't merge back in "example" default
        return super(ClassifierCollection, cls).from_config(
            {'classifiers': classifier_map},
            merge_default=False
        )

    def __init__(self, classifiers=None, **labeled_classifiers):
        """
        :param classifiers: Optional dictionary of semantic label keys and
            Classifier instance values.
        :type classifiers: dict[str, Classifier]

        :param labeled_classifiers: Key-word arguments may be provided where
            the key used is considered the semantic label of the provided
            Classifier instance.
        :type labeled_classifiers: Classifier

        """
        self._label_to_classifier_lock = threading.RLock()
        self._label_to_classifier = {}

        # Go though classifiers map and key-word arguments, check that values
        # are actually classifiers.
        if classifiers is not None:
            for label, classifier in classifiers.items():
                if not isinstance(classifier, Classifier):
                    raise ValueError("Found a non-Classifier instance value "
                                     "for key '%s'" % label)
                self._label_to_classifier[label] = classifier

        for label, classifier in labeled_classifiers.items():
            if not isinstance(classifier, Classifier):
                raise ValueError("Found a non-Classifier instance value "
                                 "for key '%s'" % label)
            elif label in self._label_to_classifier:
                raise ValueError("Duplicate classifier label '%s' provided "
                                 "in key-word arguments." % label)
            self._label_to_classifier[label] = classifier

    def __enter__(self):
        """
        :rtype: IqrSession
        """
        self._label_to_classifier_lock.acquire()
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._label_to_classifier_lock.release()

    def get_config(self):
        with self._label_to_classifier_lock:
            c = dict((label, to_config_dict(classifier))
                     for label, classifier
                     in self._label_to_classifier.items())
        return c

    def size(self):
        with self._label_to_classifier_lock:
            return len(self._label_to_classifier)

    __len__ = size

    def labels(self):
        """
        :return: Set of labels for currently collected classifiers.
        :rtype: set[str]
        """
        with self._label_to_classifier_lock:
            return set(self._label_to_classifier.keys())

    def add_classifier(self, label, classifier):
        """
        Add a classifier instance with associated descriptive label to this
        collection.

        :param label: String descriptive label for the classifier.
        :type label: str

        :param classifier: Classifier instance to collect.
        :type classifier: Classifier

        :raises ValueError: Classifier provided is not actually a classifier
            instance, or if the label provided already exists in this
            collection.

        :return: Self.
        :rtype: self

        """
        if not isinstance(classifier, Classifier):
            raise ValueError("Not given a Classifier instance (given type"
                             " %s)." % type(classifier))
        with self._label_to_classifier_lock:
            if label in self._label_to_classifier:
                raise ValueError("Duplicate label provided: '%s'" % label)
            self._label_to_classifier[label] = classifier
        return self

    def get_classifier(self, label):
        """
        Get the classifier instance for a given label.

        :param label: Label of the classifier to get.
        :type label: str

        :raises KeyError: No classifier for the given label.

        :return: Classifier instance.
        :rtype: Classifier

        """
        with self._label_to_classifier_lock:
            return self._label_to_classifier[label]

    def remove_classifier(self, label):
        """
        Remove a label-classifier pair from this collection.

        :param label: Label of the classifier to remove.
        :type label: str

        :raises KeyError: The given label does not reference a classifier in
            this collection.

        :return: Self.
        :rtype: self

        """
        with self._label_to_classifier_lock:
            del self._label_to_classifier[label]

    def labels_to_classifiers(self, labels=None):
        """
        Get a shallow copy mapping of classifiers for the labels given, or for
        all classifiers if no labels were explicitly given.

        This method is thread-safe and the returned dictionary is separate from
        this class's control. However, the classifier instances are still
        shared.

        :param None | Iterable[str] labels:
            One or more labels of stored classifiers to retrieve.
            If None, we will consider all stored classifiers.

        :raises MissingLabelError: Thrown when one or more labels provided do
            not associate to any currently stored classifiers.

        :return: Dictionary mapping string labels to Classifier instances.
        :rtype: dict[str, Classifier]
        """
        with self._label_to_classifier_lock:
            if labels is not None:
                labels = list(labels)
                # If we're missing some of the requested labels, complain
                missing_labels = set(labels) - self.labels()
                if missing_labels:
                    raise MissingLabelError(missing_labels)
                label2classifier = {label: self._label_to_classifier[label]
                                    for label in labels}
            else:
                label2classifier = dict(self._label_to_classifier)
        return label2classifier

    def classify(self, descriptor, labels=None,
                 factory=DFLT_CLASSIFIER_FACTORY, overwrite=False):
        """
        Apply all stored classifiers to the given descriptor element.

        We return a dictionary mapping the label of a stored classifier to the
        classifier element result produced by that classifier via the
        provided classification element factory.

        :param descriptor: Descriptor element to classify.
        :type descriptor: smqtk.representation.DescriptorElement

        :param labels: One or more labels of stored classifiers to use for
            classifying the given descriptor.  If None, use all stored
            classifiers.
        :type labels: Iterable[str]

        :param factory: Classification element factory.
        :type factory: ClassificationElementFactory

        :param overwrite: Force re-computation of the classification of the
            input descriptor.
        :type overwrite: bool

        :raises smqtk.exceptions.MissingLabelError: Some or all of the
            requested labels are missing.

        :return: Result dictionary of classifier labels to classification
            elements.
        :rtype: dict[str, smqtk.representation.ClassificationElement]

        """
        d_classifications = {}
        label2classifier = self.labels_to_classifiers(labels)
        for label, classifier in label2classifier.items():
            d_classifications[label] = classifier.classify_one_element(
                descriptor, factory=factory, overwrite=overwrite
            )
        return d_classifications

    def classify_arrays(self, array_seq, labels=None):
        """
        Apply all stored classifiers to the given iterable or matrix of numpy
        arrays.

        We return a dictionary mapping the label of a stored classifier to the
        class-confidence map result produced by that classifier's
        `classify_arrays` method.

        :param Sequence[np.ndarray] array_seq:
            Sequence of descriptor vectors, as numpy arrays, to be classified.
        :param Sequence[str] labels:
            One or more labels of stored classifiers to use for classifying the
            given descriptors.  If None (the default), use all stored
            classifiers.

        :raises MissingLabelError: Thrown when one or more labels provided do
            not associate to any currently stored classifiers.

        :return: Dictionary of result predictions for each classifier.
        :rtype: dict[str, list[dict[collections.abc.Hashable, float]]]
        """
        label2classifier = self.labels_to_classifiers(labels)
        label2pred = {}
        for label, classifier in label2classifier.items():
            label2pred[label] = list(classifier.classify_arrays(array_seq))
        return label2pred
