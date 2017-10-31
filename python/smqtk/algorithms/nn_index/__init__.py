"""
Interface for generic element-wise nearest-neighbor computation.
"""

import abc
import itertools
import os

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils.plugin import get_plugins


class NearestNeighborsIndex (SmqtkAlgorithm):
    """
    Common interface for descriptor-based nearest-neighbor computation over a
    built index of descriptors.

    Implementations, if they allow persistent storage of their index, should
    take the necessary parameters at construction time. Persistent storage
    content should be (over)written ``build_index`` is called.

    """

    def __len__(self):
        return self.count()

    @staticmethod
    def _empty_iterable_exception():
        """
        Create the exception instance to be thrown when no descriptors are
        provided to ``build_index``/``update_index``.

        :return: ValueError instance to be thrown.
        :rtype: ValueError

        """
        return ValueError("No DescriptorElement instances in provided "
                          "iterable.")

    def _check_empty_iterable(self, iterable, callback):
        """
        Check that the given iterable is not empty, then call the given callback
        function with the reconstructed iterable when it is not empty.

        :param iterable: Iterable to check.
        :type iterable: collections.Iterable

        :param callback: Function to call with the reconstructed, not-empty
            iterable.
        :type callback: (collections.Iterable) -> None

        """
        i = iter(iterable)
        try:
            first = next(i)
        except StopIteration:
            raise self._empty_iterable_exception()
        callback(itertools.chain([first], i))

    def build_index(self, descriptors):
        """
        Build the index with the given descriptor data elements.

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        self._check_empty_iterable(descriptors, self._build_index)

    def update_index(self, descriptors):
        """
        Additively update the current index with the one or more descriptor
        elements given.

        If no index exists yet, a new one should be created using the given
        descriptors.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to add to this
            index.
        :type descriptors: collections.Iterable[smqtk.representation
                                                     .DescriptorElement]

        """
        self._check_empty_iterable(descriptors, self._update_index)

    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :raises ValueError: Input query descriptor ``d`` has no vector set.
        :raises ValueError: Current index is empty.

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
        return self._nn(d, n)

    @abc.abstractmethod
    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """

    @abc.abstractmethod
    def _build_index(self, descriptors):
        """
        Internal method to be implemented by sub-classes to build the index with
        the given descriptor data elements.

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """

    @abc.abstractmethod
    def _update_index(self, descriptors):
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more descriptor elements given.

        If no index exists yet, a new one should be created using the given
        descriptors.

        :param descriptors: Iterable of descriptor elements to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """

    @abc.abstractmethod
    def _nn(self, d, n=1):
        """
        Internal method to be implemented by sub-classes to return the nearest
        `N` neighbors to the given descriptor element.

        When this internal method is called, we have already checked that there
        is a vector in ``d`` and our index is not empty.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """


def get_nn_index_impls(reload_modules=False):
    """
    Discover and return discovered ``NearestNeighborsIndex`` classes. Keys in
    the returned map are the names of the discovered classes, and the paired
    values are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``NN_INDEX_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``NN_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``NearestNeighborsIndex``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "NN_INDEX_PATH"
    helper_var = "NN_INDEX_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var,
                       NearestNeighborsIndex, reload_modules=reload_modules)
