"""
Interface for generic element-wise nearest-neighbor computation.
"""

import abc

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import check_empty_iterable


class NearestNeighborsIndex (SmqtkAlgorithm):
    """
    Common interface for descriptor-based nearest-neighbor computation over a
    built index of descriptors.

    Implementations, if they allow persistent storage of their index, should
    take the necessary parameters at construction time. Persistent storage
    content should be (over)written ``build_index`` is called.

    Implementations should be thread safe and appropriately protect
    internal model components from concurrent access and modification.

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
        check_empty_iterable(descriptors, self._build_index,
                             self._empty_iterable_exception())

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
        check_empty_iterable(descriptors, self._update_index,
                             self._empty_iterable_exception())

    def remove_from_index(self, uids):
        """
        Partially remove descriptors from this index associated with the given
        UIDs.

        :param uids: Iterable of UIDs of descriptors to remove from this index.
        :type uids: collections.Iterable[collections.Hashable]

        :raises ValueError: No data available in the given iterable.
        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.  The index should not be modified.

        """
        check_empty_iterable(uids, self._remove_from_index,
                             self._empty_iterable_exception())

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
    def _remove_from_index(self, uids):
        """
        Internal method to be implemented by sub-classes to partially remove
        descriptors from this index associated with the given UIDs.

        :param uids: Iterable of UIDs of descriptors to remove from this index.
        :type uids: collections.Iterable[collections.Hashable]

        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.

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
