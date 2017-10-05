import os

from smqtk.algorithms.nn_index.hash_index import HashIndex
from smqtk.utils.vptree import vp_make_tree, vp_knn_recursive
from smqtk.utils.bit_utils import (
    bit_vector_to_int_large,
    int_to_bit_vector_large,
)
from smqtk.utils.metrics import hamming_distance


__author__ = "william.p.hicks@gmail.com"


class VPTreeHashIndex (HashIndex):
    """
    Index using a basic vantage point tree
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, file_cache=None, random_seed=None):
        """
        Initialize vantage point tree hash index

        :param file_cache: Optional path to a file to cache our index to.
        :type file_cache: str

        """
        super(VPTreeHashIndex, self).__init__()
        self.file_cache = file_cache
        self.random_seed = random_seed
        self.vpt = None
        self.load_cache()

    def get_config(self):
        return {
            'file_cache': self.file_cache,
            'random_seed': self.random_seed,
        }

    def load_cache(self):
        """
        Load from file cache if we have one
        """
        if self.file_cache and os.path.isfile(self.file_cache):
            raise NotImplementedError

    def save_cache(self):
        """
        save to file cache if configures
        """
        if self.file_cache:
            raise NotImplementedError

    def count(self):
        if self.vpt is not None:
            return self.vpt.num_descendants + 1
        else:
            return 0

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
        hashes_as_ints = map(bit_vector_to_int_large, hashes)
        if not hashes_as_ints:
            raise ValueError("No hashes given to index.")
        self.vpt = vp_make_tree(hashes_as_ints, hamming_distance,
                                r_seed=self.random_seed)
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
        super(VPTreeHashIndex, self).nn(h, n)
        h_int = bit_vector_to_int_large(h)
        neighbors, dists = vp_knn_recursive(
            h_int, n, self.vpt, hamming_distance)
        bits = len(h)
        neighbors = [
            int_to_bit_vector_large(neighbor, bits=bits)
            for neighbor in neighbors
        ]
        return neighbors, dists
