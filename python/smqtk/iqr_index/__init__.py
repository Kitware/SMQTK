import abc
import logging

from smqtk.utils.configurable_interface import Configurable


__author__ = 'purg'


class IqrIndex (Configurable):
    """
    Abstract class for IQR index implementations.

    Similar to a traditional nearest-neighbors algorthm, An IQR index provides a specialized
    nearest-neighbors interface that can take multiple examples of positively
    and negatively relevant exemplars in order to produce a [0, 1] ranking of
    the indexed elements by determined relevancy.

    """
    __metaclass__ = abc.ABCMeta

    def __len__(self):
        return self.count()

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    # noinspection PyMethodParameters
    @abc.abstractmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.

        Since certain implementations may require additional dependencies that
        may not yet be available on the system, this method should check for
        those dependencies and return a boolean saying if the implementation is
        usable.

        NOTES:
            - This should be a class method
            - When not available, this should emit a warning message pointing to
                documentation on how to get/install required dependencies.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """

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
        :type descriptors: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """
        return

    @abc.abstractmethod
    def rank(self, pos, neg=()):
        """
        Rank the currently indexed elements given ``pos`` positive and ``neg``
        negative exemplar descriptor elements.

        :param pos: Iterable of positive exemplar DescriptorElement instances.
        :type pos: collections.Iterable[smqtk.data_rep.DescriptorElement]

        :param neg: Optional iterable of negative exemplar DescriptorElement
            instances.
        :type neg: collections.Iterable[smqtk.data_rep.DescriptorElement]

        :return: Map of descriptor UUID to rank value within [0, 1] range, where
            a 1.0 means most relevant and 0.0 meaning least relevant.
        :rtype: dict[collections.Hashable, float]

        """
        return


def get_iqr_index():
    """
    Discover and return SimilarityNN implementation classes found in the given
    plugin search directory. Keys in the returned map are the names of the
    discovered classes, and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``IQR_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same name and casing as the module. If
    neither are found, the module is skipped.

    :return: Map of discovered class object of type ``SimilarityNN`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    import os
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "IQR_INDEX_CLASS"

    def class_filter(cls):
        log = logging.getLogger('.'.join([__name__, 'get_iqr_index',
                                          'class_filter']))
        if not cls.is_usable():
            log.warn("Class type '%s' not usable, filtering out.",
                     cls.__name__)
            return False
        return True

    return get_plugins(__name__, this_dir, helper_var, IqrIndex,
                       class_filter)
