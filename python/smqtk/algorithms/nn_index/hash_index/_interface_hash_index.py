import abc

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import check_empty_iterable


class HashIndex (SmqtkAlgorithm):
    """
    Specialized ``NearestNeighborsIndex`` for indexing unique hash codes
    bit-vectors) in memory (numpy arrays) using the hamming distance metric.

    Implementations of this interface cannot be used in place of something
    requiring a ``NearestNeighborsIndex`` implementation due to the speciality
    of this interface.

    Only unique bit vectors should be indexed. The ``nn`` method should not
    return the same bit vector more than once for any query.
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
        return ValueError("No hash vectors in provided iterable.")

    def build_index(self, hashes):
        """
        Build the index with the given hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :raises ValueError: No data available in the given iterable.

        :param hashes: Iterable of descriptor elements to build index
            over.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """
        check_empty_iterable(hashes, self._build_index,
                             self._empty_iterable_exception())

    def update_index(self, hashes):
        """
        Additively update the current index with the one or more hash vectors
        given.

        If no index exists yet, a new one should be created using the given hash
        vectors.

        :raises ValueError: No data available in the given iterable.

        :param hashes: Iterable of numpy boolean hash vectors to add to this
            index.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """
        check_empty_iterable(hashes, self._update_index,
                             self._empty_iterable_exception())

    def remove_from_index(self, hashes):
        """
        Partially remove hashes from this index.

        :param hashes: Iterable of numpy boolean hash vectors to remove from
            this index.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        :raises ValueError: No data available in the given iterable.
        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.

        """
        check_empty_iterable(hashes, self._remove_from_index,
                             self._empty_iterable_exception())

    def nn(self, h, n=1):
        """
        Return the nearest `N` neighbor hash codes as bit-vectors to the given
        hash code bit-vector.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query (normalized hamming distance).

        :raises ValueError: Current index is empty.

        :param h: Hash code to compute the neighbors of. Should be the same bit
            length as indexed hash codes.
        :type h: numpy.ndarray[bool]

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.
        :rtype: (tuple[numpy.ndarray[bool]], tuple[float])

        """
        # Only check for count because we're no longer dealing with descriptor
        # elements.
        if not self.count():
            raise ValueError("No index currently set to query from!")
        return self._nn(h, n)

    @abc.abstractmethod
    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        pass

    @abc.abstractmethod
    def _build_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to build the index with
        the given hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :param hashes: Iterable of descriptor elements to build index
            over.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """

    @abc.abstractmethod
    def _update_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more hash vectors given.

        If no index exists yet, a new one should be created using the given hash
        vectors.

        :param hashes: Iterable of numpy boolean hash vectors to add to this
            index.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """

    @abc.abstractmethod
    def _remove_from_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to partially remove
        hashes from this index.

        :param hashes: Iterable of numpy boolean hash vectors to remove from
            this index.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        :raises KeyError: One or more hashes provided do not match any stored
            hashes.  The index should not be modified.

        """

    @abc.abstractmethod
    def _nn(self, h, n=1):
        """
        Internal method to be implemented by sub-classes to return the nearest
        `N` neighbor hash codes as bit-vectors to the given hash code
        bit-vector.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query (normalized hamming distance).

        When this internal method is called, we have already checked that our
        index is not empty.

        :param h: Hash code to compute the neighbors of. Should be the same bit
            length as indexed hash codes.
        :type h: numpy.ndarray[bool]

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.
        :rtype: (tuple[numpy.ndarray[bool]], tuple[float])

        """
