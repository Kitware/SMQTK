"""
Interface for generic element-wise nearest-neighbor computation.
"""
__author__ = 'purg'

import abc
import logging


class SimilarityNN (object):
    """
    Common interface for content-based nearest-neighbor computation.

    For what we're trying to go for as being a simplified API, base DataElement
    instances are used when interacting with these classes. Thus, since
    similarity is computed via descriptors, a SimilarityNN instance must be
    constructed with a ContentDescriptor instance that it should use for feature
    computation. This being the case, if the ContentDescriptor implementation
    being used does not have caching mechanisms, redundant descriptor
    computation may occur with respect to what else may be going on in the
    system.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, content_descriptor):
        """
        Initialize this similarity indexer to compute similarity based on the
        given content descriptor.

        :param content_descriptor: Content descriptor instance to provide
            descriptors
        :type content_descriptor: smqtk.content_description.ContentDescriptor

        """
        self._content_descriptor = content_descriptor

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    @abc.abstractmethod
    def is_usable(self):
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
        return

    @abc.abstractmethod
    def build_index(self, data):
        """
        Build the index over the given data elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        :raises ValueError: No data available in the given iterable.

        :param data: Iterable of data elements to build index over.
        :type data: collections.Iterable[smqtk.data_rep.DataElement]

        """
        return

    @abc.abstractmethod
    def add_to_index(self, data):
        """
        Add the given data element to the index.

        :param data: New data element to add to our index.
        :type data: smqtk.data_rep.DataElement

        """
        return

    @abc.abstractmethod
    def nn(self, d, N=1):
        """
        Return the nearest N neighbors to the given data element.

        """
        return


def get_similarity_nn():
    """
    Discover and return SimilarityNN implementation classes found in the given
    plugin search directory. Keys in the returned map are the names of the
    discovered classes, and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``SIMILARITY_NN_CLASS``, which can either be a single class object or
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
    helper_var = "SIMILARITY_NN_CLASS"

    def class_filter(cls):
        log = logging.getLogger('.'.join([__name__, 'get_similarity_nn',
                                          'class_filter']))
        if not cls.is_usable():
            log.warn("Class type '%s' not usable, filtering out...",
                     cls.__name__)
            return False
        return True

    return get_plugins(__name__, this_dir, helper_var, SimilarityNN,
                       class_filter)
