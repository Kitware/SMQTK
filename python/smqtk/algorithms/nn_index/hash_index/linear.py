import heapq
import os

import numpy

from smqtk.algorithms.nn_index.hash_index import HashIndex
from smqtk.utils.bit_utils import (
    bit_vector_to_int_large,
    int_to_bit_vector_large,
)
from smqtk.utils.distance_functions import hamming_distance


__author__ = "paul.tunison@kitware.com"


class LinearHashIndex (HashIndex):
    """
    Basic linear index using heap sort.
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, file_cache=None):
        """
        Initialize linear, brute-force hash index

        :param file_cache: Optional path to a file to cache our index to.
        :type file_cache: str

        """
        super(LinearHashIndex, self).__init__()
        self.file_cache = file_cache
        self.index = numpy.array([], bool)
        self.load_cache()

    def get_config(self):
        return {
            'file_cache': self.file_cache,
        }

    def load_cache(self):
        """
        Load from file cache if we have one
        """
        if self.file_cache and os.path.isfile(self.file_cache):
            self.index = numpy.load(self.file_cache)

    def save_cache(self):
        """
        save to file cache if configures
        """
        if self.file_cache:
            numpy.save(self.file_cache, self.index)

    def count(self):
        return len(self.index)

    def build_index(self, hashes):
        """
        Build the index with the give hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the index, not add to
        it, or raise an exception to as to protect the current index.

        :raises ValueError: No data available in the given iterable.

        :param hashes: Iterable of descriptor elements to build index
            over.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """
        new_index = numpy.array(map(bit_vector_to_int_large, hashes))
        if not new_index.size:
            raise ValueError("No hashes given to index.")
        self.index = new_index
        self.save_cache()

    def nn(self, h, n=1):
        """
        Return the nearest `N` neighbors to the given hash code.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query.

        :param h: Hash code to compute the neighbors of. Should be the same bit
            length as indexed hash codes.
        :type h: numpy.ndarray[bool]

        :param n: Number of nearest neighbors to find.
        :type n: int

        :raises ValueError: No index to query from.

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.
        :rtype: (tuple[numpy.ndarray[bool], tuple[float])

        """
        super(LinearHashIndex, self).nn(h, n)

        h_int = bit_vector_to_int_large(h)
        bits = len(h)
        #: :type: list[int|long]
        near_codes = \
            heapq.nsmallest(n, self.index,
                            lambda e: hamming_distance(h_int, e)
                            )
        distances = map(hamming_distance, near_codes,
                        [h_int] * len(near_codes))
        return [int_to_bit_vector_large(c, bits) for c in near_codes], \
               [d / float(bits) for d in distances]
