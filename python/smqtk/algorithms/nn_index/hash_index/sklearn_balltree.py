import os

import numpy

from smqtk.algorithms.nn_index.hash_index import HashIndex

try:
    from sklearn.neighbors import BallTree, DistanceMetric
except ImportError:
    BallTree = None
    DistanceMetric = None


__author__ = "paul.tunison@kitware.com"


class SkLearnBallTreeHashIndex (HashIndex):
    """
    Hash index using the ball tree implementation in scikit-learn.

    *Note:* **When saving this object's model or pickling, we do not naively
    pickle the underlying ball tree due to issues when saving the state of a
    large ball tree. We instead get the state and split its contents up for
    separate serialization via known safe methods.**
    """

    @classmethod
    def is_usable(cls):
        return BallTree is not None

    def __init__(self, file_cache=None, leaf_size=40, random_seed=None):
        """
        Initialize Scikit-Learn BallTree index for hash codes.

        :param file_cache: Optional path to a file to cache our index to. This
            must have the `.npz` suffix.
        :type file_cache: str

        :param leaf_size: Number of points at which to switch to brute-force.
        :type leaf_size: int

        :param random_seed: Optional random number generator seed (numpy).
        :type random_seed: None | int

        """
        super(SkLearnBallTreeHashIndex, self).__init__()
        self.file_cache = file_cache
        self.leaf_size = leaf_size
        self.random_seed = random_seed

        if self.file_cache and not self.file_cache.endswith('.npz'):
            raise ValueError("File cache path given does not specify and npz "
                             "file.")

        # the actual index
        #: :type: sklearn.neighbors.BallTree
        self.bt = None

        self.load_model()

    def save_model(self):
        if self.file_cache and self.bt:
            self._log.debug("Saving model: %s", self.file_cache)
            # Saving BT component matrices in separate files using numpy.save
            # - Not saving distance function because its always going to be
            #   hamming distance (see ``build_index``).
            s = self.bt.__getstate__()
            tail = s[4:11]
            numpy.savez(self.file_cache,
                        data_arr=s[0],
                        idx_array_arr=s[1],
                        node_data_arr=s[2],
                        node_bounds_arr=s[3],
                        tail=tail)
            self._log.debug("Saving model: Done")

    def load_model(self):
        if self.file_cache and os.path.isfile(self.file_cache):
            self._log.debug("Loading mode: %s", self.file_cache)
            with numpy.load(self.file_cache) as cache:
                tail = tuple(cache['tail'])
                s = (cache['data_arr'], cache['idx_array_arr'],
                     cache['node_data_arr'], cache['node_bounds_arr']) +\
                    tail + (DistanceMetric.get_metric('hamming'),)
            #: :type: sklearn.neighbors.BallTree
            self.bt = BallTree.__new__(BallTree)
            self.bt.__setstate__(s)
            self._log.debug("Loading mode: Done")

    def get_config(self):
        return {
            'file_cache': self.file_cache,
            'leaf_size': self.leaf_size,
            'random_seed': self.random_seed,
        }

    def count(self):
        return self.bt.data.shape[0] if self.bt else 0

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
        self._log.debug("Building ball tree")
        if self.random_seed is not None:
            numpy.random.seed(self.random_seed)
        # BallTree can't take iterables, so catching input in a list first
        hash_list = list(hashes)
        if not hash_list:
            raise ValueError("No hashes given.")

        # If distance metric ever changes, need to update save/load model
        # functions.
        self.bt = BallTree(hash_list, self.leaf_size, metric='hamming')
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
        :rtype: (tuple[numpy.ndarray[bool]], tuple[float])

        """
        super(SkLearnBallTreeHashIndex, self).nn(h, n)
        # Reselect N based on how many hashes are currently indexes
        n = min(n, self.count())
        # Reshaping ``h`` into an array of arrays, with just one array (ball
        # tree deprecation warns when giving it a single array).
        dists, idxs = self.bt.query([h], n, return_distance=True)
        # only indexing the first entry became we're only querying with one
        # vector
        neighbors = numpy.asarray(self.bt.data)[idxs[0]].astype(bool)
        return neighbors, dists[0]
