import abc
import os

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils.plugin import get_plugins


__author__ = "paul.tunison@kitware.com"


class RelevancyIndex (SmqtkAlgorithm):
    """
    Abstract class for IQR index implementations.

    Similar to a traditional nearest-neighbors algorithm, An IQR index provides
    a specialized nearest-neighbors interface that can take multiple examples of
    positively and negatively relevant exemplars in order to produce a [0, 1]
    ranking of the indexed elements by determined relevancy.

    """

    def __len__(self):
        return self.count()

    @abc.abstractmethod
    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """

    @abc.abstractmethod
    def build_index(self, descriptors):
        """
        Build the index based on the given iterable of descriptor elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors: collections.Iterable[smqtk.representation.DescriptorElement]

        """

    @abc.abstractmethod
    def rank(self, pos, neg):
        """
        Rank the currently indexed elements given ``pos`` positive and ``neg``
        negative exemplar descriptor elements.

        :param pos: Iterable of positive exemplar DescriptorElement instances.
            This may be optional for some implementations.
        :type pos: collections.Iterable[smqtk.representation.DescriptorElement]

        :param neg: Iterable of negative exemplar DescriptorElement instances.
            This may be optional for some implementations.
        :type neg: collections.Iterable[smqtk.representation.DescriptorElement]

        :return: Map of indexed descriptor elements to a rank value between
            [0, 1] (inclusive) range, where a 1.0 means most relevant and 0.0
            meaning least relevant.
        :rtype: dict[smqtk.representation.DescriptorElement, float]

        """


def get_relevancy_index_impls(reload_modules=False):
    """
    Discover and return discovered ``RelevancyIndex`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``RELEVANCY_INDEX_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``RELEVANCY_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``RelevancyIndex``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "RELEVANCY_INDEX_PATH"
    helper_var = "RELEVANCY_INDEX_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var, RelevancyIndex,
                       reload_modules=reload_modules)
