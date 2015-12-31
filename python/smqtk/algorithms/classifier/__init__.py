import abc
import os

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils.plugin import get_plugins


__author__ = 'paul.tunison@kitware.com, jacob.becker@kitware.com'


class Classifier (SmqtkAlgorithm):
    """
    Interface for algorithms that classify input descriptors into discrete
    labels and/or label confidences.
    """

    def classify(self, d, factory, overwrite=False):
        """
        Classify the input descriptor against one or more discrete labels,
        outputting a ClassificationElement containing the classification result.


        We return confidence values for each label the configured model
        contains. Implementations may act in a discrete manner whereby only one
        label is marked with a ``1`` value (others being ``0``), or in a
        continuous manner whereby each label is given a confidence-like value in
        the [0, 1] range.

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


def get_classifier_impls(reload_modules=False):
    """
    Discover and return discovered ``Classifier`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``CLASSIFIER_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``CLASSIFIER_CLASS``, which can either be a single class object or an
    iterable of class objects, to be specifically exported. If the variable is
    set to None, we skip that module and do not import anything. If the variable
    is not present, we look at attributes defined in that module for classes
    that descend from the given base class type. If none of the above are found,
    or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``Classifier``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "CLASSIFIER_PATH"
    helper_var = "CLASSIFIER_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var, Classifier,
                       reload_modules=reload_modules)
