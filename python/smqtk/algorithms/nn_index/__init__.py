"""
Interface for generic element-wise nearest-neighbor computation.
"""

import abc
import logging

from smqtk.algorithms import SmqtkAlgorithm


__author__ = "paul.tunison@kitware.com"


class NearestNeighborsIndex (SmqtkAlgorithm):
    """
    Common interface for descriptor-based nearest-neighbor computation over a
    built index of descriptors.

    Implementations, if they allow persistent storage of their index, should
    take the necessary parameters at construction time. Persistant storage
    content should be (over)written ``build_index`` is called.

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
        Build the index over the descriptor data elements.

        Subsequent calls to this method should rebuild the index, not add to it,
        or raise an exception to as to protect the current index.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors: collections.Iterable[smqtk.representation.DescriptorElement]

        """

    @abc.abstractmethod
    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """
        if not d.has_vector():
            raise ValueError("Query descriptor did not have a vector set!")
        elif not self.count():
            raise ValueError("No index currently set to query from!")


def get_nn_index_impls(reload_modules=False):
    """
    Discover and return ``NearestNeighborsIndex`` implementation classes found
    in the given plugin search directory. Keys in the returned map are the names
    of the discovered classes, and the paired values are the actual class type
    objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``NN_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same name and casing as the module. If
    neither are found, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``NearestNeighborsIndex``
        whose keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    import os
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "NN_INDEX_CLASS"
    return get_plugins(__name__, this_dir, helper_var, NearestNeighborsIndex,
                       reload_modules)
