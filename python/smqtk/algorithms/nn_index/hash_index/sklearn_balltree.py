import threading

import numpy as np
from six import BytesIO, next
from six.moves import map
from sklearn.neighbors import BallTree, DistanceMetric

from smqtk.algorithms.nn_index.hash_index import HashIndex
from smqtk.representation import get_data_element_impls
from smqtk.utils import merge_dict, plugin


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

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as
        arguments, turning those argument names into configuration dictionary
        keys. If any of those arguments have defaults, we will add those values
        into the configuration dictionary appropriately. The dictionary
        returned should only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this
        class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        c = super(SkLearnBallTreeHashIndex, cls).get_default_config()
        c['cache_element'] = plugin.make_config(get_data_element_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless an instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: SkLearnBallTreeHashIndex

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        # Parse ``cache_element`` configuration if set.
        cache_element = None
        if config_dict['cache_element'] and \
                config_dict['cache_element']['type']:
            cache_element = \
                plugin.from_plugin_config(config_dict['cache_element'],
                                          get_data_element_impls())
        config_dict['cache_element'] = cache_element

        return super(SkLearnBallTreeHashIndex, cls).from_config(config_dict,
                                                                False)

    def __init__(self, cache_element=None, leaf_size=40, random_seed=None):
        """
        Initialize Scikit-Learn BallTree index for hash codes.

        :param cache_element: Optional data element to cache our index to.
        :type cache_element: smqtk.representation.DataElement | None

        :param leaf_size: Number of points at which to switch to brute-force.
        :type leaf_size: int

        :param random_seed: Optional random number generator seed (numpy).
        :type random_seed: None | int

        """
        super(SkLearnBallTreeHashIndex, self).__init__()
        self.cache_element = cache_element
        self.leaf_size = leaf_size
        self.random_seed = random_seed

        self._model_lock = threading.RLock()

        # the actual index
        #: :type: sklearn.neighbors.BallTree
        self.bt = None

        self.load_model()

    def get_config(self):
        c = merge_dict(self.get_default_config(), {
            'leaf_size': self.leaf_size,
            'random_seed': self.random_seed,
        })
        if self.cache_element:
            c['cache_element'] = merge_dict(c['cache_element'],
                                            plugin.to_plugin_config(
                                                self.cache_element))
        return c

    def save_model(self):
        """
        Cache a built B-Tree index to the configured cache element. This only
        occurs if we have a non-null cache element and a btree to save.

        :raises ValueError: If the cache element configured is not writable.

        """
        with self._model_lock:
            if self.cache_element:
                if self.cache_element.is_read_only():
                    raise ValueError("Configured cache element (%s) is read-"
                                     "only." % self.cache_element)
                if self.bt is not None:
                    self._log.debug("Saving model: %s", self.cache_element)
                    # Saving BT component matrices separately.
                    # - Not saving distance function because its always going
                    #   to be hamming distance (see ``build_index``).
                    s = self.bt.__getstate__()
                    # Everything after known numpy ndarrays we consider as a
                    # chunk.  Importantly, index 11 should be the distance
                    # metric function used in the model.  We currently assume
                    # its a hamming distance metric.
                    tail = s[4:]
                    buff = BytesIO()
                    # noinspection PyTypeChecker
                    np.savez(buff,
                             data_arr=s[0],
                             idx_array_arr=s[1],
                             node_data_arr=s[2],
                             node_bounds_arr=s[3],
                             tail=tail)
                    self.cache_element.set_bytes(buff.getvalue())
                    self._log.debug("Saving model: Done")
                else:
                    # No ball tree, empty cache.
                    self.cache_element.set_bytes('')

    def load_model(self):
        """
        Load a btree index from the configured cache element. This only occurs
        if there is a cache element configured and there are bytes there to
        read.
        """
        with self._model_lock:
            if self.cache_element and not self.cache_element.is_empty():
                self._log.debug("Loading model from cache: %s",
                                self.cache_element)
                buff = BytesIO(self.cache_element.get_bytes())
                with np.load(buff) as cache:
                    tail = tuple(cache['tail'])
                    s = [cache['data_arr'], cache['idx_array_arr'],
                         cache['node_data_arr'], cache['node_bounds_arr']]
                    s.extend(tail)
                    s[11] = DistanceMetric.get_metric('hamming')
                    s = tuple(s)
                # noinspection PyTypeChecker
                #: :type: sklearn.neighbors.BallTree
                self.bt = BallTree.__new__(BallTree)
                self.bt.__setstate__(s)
                self._log.debug("Loading mode: Done")

    def count(self):
        with self._model_lock:
            return self.bt.data.shape[0] if self.bt else 0

    def _build_bt_internal(self, vec_list):
        """
        Internal shared BallTree build function given a list of boolean hash
        vectors.

        This is here to avoid calling `_build_index` from `_update...` and
        `_remove...` which cuts down on redundant data consistency checking.

        :param vec_list: List of bool-valued numpy arrays.
        :type vec_list: list[np.ndarray[bool]] | np.ndarray[np.ndarray[bool]]

        """
        if len(vec_list) > 0:
            # If distance metric ever changes, need to update save/load model
            # functions.
            self.bt = BallTree(vec_list, self.leaf_size,
                               metric='hamming')
        else:
            self.bt = None
        self.save_model()

    def _build_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to build the index
        with the given hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :param hashes: Iterable of descriptor elements to build index
            over.
        :type hashes: collections.Iterable[np.ndarray[bool]]
        """
        with self._model_lock:
            self._log.debug("Building ball tree")
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            # BallTree can't take iterables, so catching input in a set of
            # tuples first in order to cull out duplicates (BT will index
            # duplicate values happily).
            hash_tuple_set = set(map(lambda v: tuple(v), hashes))
            # Convert tuples back into numpy arrays for BallTree constructor.
            hash_vector_list = list(map(lambda t: np.array(t), hash_tuple_set))
            self._build_bt_internal(hash_vector_list)

    def _update_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more hash vectors given.

        If no index exists yet, a new one should be created using the given
        hash vectors.

        *Note:* The scikit-learn ball-tree implementation does not support
        incremental updating of its model, so we need to rebuild the model from
        scratch using the currently indexed hashes and the new ones provided to
        this method.

        :param hashes: Iterable of numpy boolean hash vectors to add to this
            index.
        :type hashes: collections.Iterable[np.ndarray[bool]]

        """
        with self._model_lock:
            # Can't use iterators with numpy operations.
            new_hashes = tuple(hashes)
            if self.bt is None:
                # 0-row array using bit-vector size of first new entry length.
                # - Must have at least one new hash due to super-method check.
                indexed_hash_vectors = np.ndarray((0, len(new_hashes[0])))
            else:
                indexed_hash_vectors = self.bt.data
            # Build a new index as normal with the union of source data.
            self._log.debug("Updating index by rebuilding with union.")
            self._build_bt_internal(
                np.concatenate([indexed_hash_vectors, new_hashes], 0)
            )

    def _remove_from_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to partially remove
        hashes from this index.

        *Note:* The scikit-learn ball-tree implementation does not support
        incremental removal from its model, so we need to rebuild the model
        from scratch using the currently indexed hashes minus the ones provided
        to this method.

        :param hashes: Iterable of numpy boolean hash vectors to remove from
            this index.
        :type hashes: collections.Iterable[np.ndarray[bool]]

        :raises KeyError: One or more hashes provided do not match any stored
            hashes.  The index should not be modified.

        """
        with self._model_lock:
            # Convert to a set of hashable tuples for removal
            if self.bt:
                tuple_matrix = set(tuple(r) for r in
                                   np.asarray(self.bt.data).tolist())
                hash_tuples = set()
                for h in hashes:
                    # this is oddly faster than tuple(h)
                    h_t = tuple(h.tolist())
                    if h_t not in tuple_matrix:
                        raise KeyError(h)
                    hash_tuples.add(h_t)
                new_data = tuple_matrix - hash_tuples
                self._build_bt_internal(
                    list(map(lambda t: np.array(t), new_data))
                )
            else:
                # No index built, so anything is a key error.
                # We can also only be here if hashes was non-zero in size.
                raise KeyError(np.asarray(next(iter(hashes))))

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
        :type h: np.ndarray[bool]

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.
        :rtype: (tuple[np.ndarray[bool]], tuple[float])

        """
        with self._model_lock:
            # Reselect N based on how many hashes are currently indexes
            n = min(n, self.count())
            # Reshaping ``h`` into an array of arrays, with just one array
            # (ball tree deprecation warns when giving it a single array).
            dists, idxs = self.bt.query([h], n, return_distance=True)
            # only indexing the first entry became we're only querying with one
            # vector
            neighbors = np.asarray(self.bt.data)[idxs[0]].astype(bool)
            return neighbors, dists[0]
