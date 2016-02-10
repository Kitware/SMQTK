import abc
import os

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import (
    bin_utils,
    parallel,
    plugin,
)


__author__ = 'paul.tunison@kitware.com, jacob.becker@kitware.com'
__all__ = [
    "Classifier",
    "SupervisedClassifier",
    "get_classifier_impls",
]


class Classifier (SmqtkAlgorithm):
    """
    Interface for algorithms that classify input descriptors into discrete
    labels and/or label confidences.
    """

    def classify(self, d, factory, overwrite=False):
        """
        Classify the input descriptor against one or more discrete labels,
        outputting a ClassificationElement containing the classification
        result.

        We return confidence values for each label the configured model
        contains. Implementations may act in a discrete manner whereby only one
        label is marked with a ``1`` value (others being ``0``), or in a
        continuous manner whereby each label is given a confidence-like value
        in the [0, 1] range.

        The returned ``ClassificationElement`` will have the same UUID as the
        input ``DescriptorElement``.

        :param d: Input descriptor to classify
        :type d: smqtk.representation.DescriptorElement

        :param factory: Classification element factory
        :type factory: smqtk.representation.ClassificationElementFactory

        :param overwrite: Recompute classification of the input descriptor and
            set the results to the ClassificationElement produced by the
            factory.
        :type overwrite: bool

        :raises RuntimeError: Could not perform classification for some reason
            (see message).

        :return: Classification result element
        :rtype: smqtk.representation.ClassificationElement

        """
        c_elem = factory.new_classification(self.name, d.uuid())
        if overwrite or not c_elem.has_classifications():
            c = self._classify(d)
            c_elem.set_classification(c)
        else:
            self._log.debug("Found existing classification in generated "
                            "element")

        return c_elem

    def classify_async(self, d_iter, factory, overwrite=False, procs=None,
                       use_multiprocessing=False, ri=None):
        """
        Asynchronously classify the DescriptorElements in the given iterable.

        :param d_iter: Iterable of DescriptorElements
        :type d_iter:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param factory: Classifier element factory to use for element
            generation.
        :type factory: smqtk.representation.ClassificationElementFactory

        :param overwrite: Recompute classification of the input descriptor and
            set the results to the ClassificationElement produced by the
            factory.
        :type overwrite: bool

        :param procs: Explicit number of cores/thread/processes to use.
        :type procs: None | int

        :param use_multiprocessing: Use ``multiprocessing.pool.Pool`` instead of
            ``multiprocessing.pool.ThreadPool``.
        :type use_multiprocessing: bool

        :param ri: Progress reporting interval in seconds. Set to a value > 0 to
            enable. Disabled by default.
        :type ri: float | None

        :return: Mapping of input DescriptorElement instances to the computed
            ClassificationElement. ClassificationElement UUID's are congruent
            with the UUID of the DescriptorElement
        :rtype: dict[smqtk.representation.DescriptorElement,
                     smqtk.representation.ClassificationElement]

        """
        self._log.info("Async classifying descriptors")
        ri = ri and ri > 0 and ri

        def work(d_elem):
            return d_elem, self.classify(d_elem, factory, overwrite)

        classifications = parallel.parallel_map(
            work, d_iter,
            cores=procs,
            ordered=False,
            use_multiprocessing=use_multiprocessing,
        )

        r_state = [0] * 7
        if ri:
            r_progress = bin_utils.report_progress
        else:
            def r_progress(*_):
                return

        d2c_map = {}
        for d, c in classifications:
            d2c_map[d] = c

            r_progress(self._log.debug, r_state, ri)

        return d2c_map

    #
    # Abstract methods
    #

    @abc.abstractmethod
    def get_labels(self):
        """
        Get the sequence of class labels that this classifier can classify
        descriptors into..

        :return: Sequence of possible classifier labels.
        :rtype: collections.Sequence[collections.Hashable]

        :raises RuntimeError: No model loaded.

        """

    @abc.abstractmethod
    def _classify(self, d):
        """
        Internal method that defines thh generation of the classification map
        for a given DescriptorElement. This returns a dictionary mapping
        integer labels to a floating point value.

        :param d: DescriptorElement containing the vector to classify.
        :type d: smqtk.representation.DescriptorElement

        :raises RuntimeError: Could not perform classification for some reason
            (see message).

        :return: Dictionary mapping trained labels to classification confidence
            values
        :rtype: dict[collections.Hashable, float]

        """


class SupervisedClassifier (Classifier):
    """
    Class of classifiers that are trainable via supervised training, i.e. are
    given specific descriptor examples for class labels (including negative
    label).
    """

    NEGATIVE_LABEL = "negative"

    @abc.abstractmethod
    def has_model(self):
        """
        :return: If this instance currently has a model loaded. If no model is
            present, classification of descriptors cannot happen (needs to be
            trained).
        :rtype: bool
        """

    @abc.abstractmethod
    def train(self, positive_classes, negatives):
        """
        Train the supervised classifier model.

        The class label ``negative`` is reserved for the negative class.

        If a model is already loaded, we will raise an exception in order to
        prevent accidental overwrite.

        NOTE:
            This abstract method provides generalized error checking and
            should be called via ``super`` in implementing methods.

        :param positive_classes: Dictionary mapping positive class labels to
            iterables of DescriptorElement training examples.
        :type positive_classes:
            dict[collections.Hashable,
                 collections.Iterable[smqtk.representation.DescriptorElement]]

        :param negatives: Iterable of negative DescriptorElement examples.
        :type negatives: collections.Iterable[smqtk.representation.DescriptorElement]

        :raises ValueError: The ``negative`` label was found in the
            ``positive_classes`` dictionary. This is reserved for the negative
            example class.
        :raises ValueError: There were no positive or negative examples.
        :raises RuntimeError: A model already exists in this instance.Following
            through with training would overwrite this model. Throwing an
            exception for information protection.

        """
        if self.has_model():
            raise RuntimeError("Instance currently has a model. Halting "
                               "training to prevent overwrite of existing "
                               "trained model.")

        if self.NEGATIVE_LABEL in positive_classes:
            raise ValueError("Found '%s' label in positive_classes map. "
                             "This label is reserved for negative class."
                             % self.NEGATIVE_LABEL)

        if not positive_classes:
            raise ValueError("No positive classes provided")
        if not negatives:
            raise ValueError("No negative examples provided.")


def get_classifier_impls(reload_modules=False, sub_interface=None):
    """
    Discover and return discovered ``Classifier`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable
          :envvar:`CLASSIFIER_PATH`
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``CLASSIFIER_CLASS``, which can either be a single class object or an
    iterable of class objects, to be specifically exported. If the variable is
    set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :param sub_interface: Only return implementations that also descend from
        the given sub-interface. The given interface must also descend from
        :class:`Classifier`.

    :return: Map of discovered class object of type :class:`Classifier`
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "CLASSIFIER_PATH"
    helper_var = "CLASSIFIER_CLASS"
    if sub_interface is None:
        base_class = Classifier
    else:
        assert issubclass(sub_interface, Classifier), \
            "The given sub-interface type must descend from `Classifier`."
        base_class = sub_interface
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var,
                              base_class, reload_modules=reload_modules)
