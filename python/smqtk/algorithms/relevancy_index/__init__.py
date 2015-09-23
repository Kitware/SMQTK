import abc
import logging

from smqtk.algorithms import SmqtkAlgorithm


__author__ = 'purg'


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
        return

    @abc.abstractmethod
    def build_index(self, descriptors):
        """
        Build the index based on the given iterable of descriptor elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors: collections.Iterable[smqtk.representation.DescriptorElement]

        """
        return

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

        :return: Map of descriptor UUID to rank value within [0, 1] range, where
            a 1.0 means most relevant and 0.0 meaning least relevant.
        :rtype: dict[collections.Hashable, float]

        """
        return


def get_relevancy_index_impls(reload_modules=False):
    """
    Discover and return ``RelevancyIndex`` implementation classes found in the
    given plugin search directory. Keys in the returned map are the names of the
    discovered classes, and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``RELEVANCY_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same name and casing as the module. If
    neither are found, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``RelevancyIndex`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    import os
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "RELEVANCY_INDEX_CLASS"

    def class_filter(cls):
        log = logging.getLogger('.'.join([__name__,
                                          'get_relevancy_index_impls',
                                          'class_filter']))
        if not cls.is_usable():
            log.warn("Class type '%s' not usable, filtering out.",
                     cls.__name__)
            return False
        return True

    return get_plugins(__name__, this_dir, helper_var, RelevancyIndex,
                       class_filter, reload_modules)
