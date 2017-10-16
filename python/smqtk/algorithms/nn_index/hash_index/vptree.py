import os
import numpy

from smqtk.algorithms.nn_index.hash_index import HashIndex
from smqtk.utils.vptree import (
    vp_make_tree, vp_knn_recursive, VpNode,
    vps_make_tree, vps_knn_recursive, VpsNode,
    vpsb_make_tree, vpsb_knn_recursive, VpsbNode
)
from smqtk.utils.bit_utils import (
    bit_vector_to_int_large,
    int_to_bit_vector_large,
)
from smqtk.utils.metrics import hamming_distance


__author__ = "william.p.hicks@gmail.com"


_tree_type_dict = {
    "vp": (vp_make_tree, vp_knn_recursive, VpNode),
    "vps": (vps_make_tree, vps_knn_recursive, VpsNode),
    "vpsb": (vpsb_make_tree, vpsb_knn_recursive, VpsbNode),
}


class VPTreeHashIndex (HashIndex):
    """
    Index using a vantage point tree
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, file_cache=None, random_seed=None, tree_type="vp"):
        """
        Initialize vantage point tree hash index

        :param file_cache: Optional path to a file to cache our index to.
        :type file_cache: str

        :param random_seed: Optional random number generator seed (numpy).
        :type random_seed: None | int

        :param tree_type: One of "vp", "vps", or "vpsb" optionally specifying
            the type of vp tree. Defaults to "vp".
        :type tree_type: str

        """
        super(VPTreeHashIndex, self).__init__()
        self.file_cache = file_cache
        self.random_seed = random_seed
        self.vpt = None
        self.tree_type = tree_type
        self._tree_builder, self._tree_searcher, self._tree_cls = \
            _tree_type_dict[self.tree_type]
        if self.file_cache and not self.file_cache.endswith('.npz'):
            raise ValueError("File cache path given does not specify an npz "
                             "file.")
        self.load_model()

    def get_config(self):
        return {
            'file_cache': self.file_cache,
            'random_seed': self.random_seed,
            'tree_type': self.tree_type,
        }

    def save_model(self):
        """
        Save to file cache if configured
        """
        if self.file_cache:
            self._log.debug("Saving model: %s", self.file_cache)
            numpy.savez(self.file_cache, **self.vpt.to_arrays())
            self._log.debug("Saving model: Done")

    def load_model(self):
        """
        Load from file cache if we have one
        """
        if self.file_cache and os.path.isfile(self.file_cache):
            self._log.debug("Loading model: %s", self.file_cache)
            with numpy.load(self.file_cache) as cache:
                self.vpt = self._tree_cls.from_arrays(**cache)

    def count(self):
        if self.vpt is not None:
            return self.vpt.num_descendants + 1
        else:
            return 0

    def build_index(self, hashes):
        """
        Build the index with the given hash codes (bit-vectors).

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
        self.vpt = self._tree_builder(hashes_as_ints, hamming_distance,
                                      r_seed=self.random_seed)
        self.save_model()

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
        neighbors, dists = self._tree_searcher(
            h_int, n, self.vpt, hamming_distance)
        bits = len(h)
        neighbors = [
            int_to_bit_vector_large(neighbor, bits=bits)
            for neighbor in neighbors
        ]
        dists = [dist_ / float(bits) for dist_ in dists]
        return neighbors, dists