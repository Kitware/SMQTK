"""
This module contains a general locality-sensitive-hashing algorithm for
nearest neighbor indexing, and various implementations of LSH functors for use
in the base.
"""
import collections

import numpy

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.algorithms.nn_index.hash_index import get_hash_index_impls
from smqtk.algorithms.nn_index.hash_index.linear import LinearHashIndex
from smqtk.algorithms.nn_index.lsh.functors import get_lsh_functor_impls
from smqtk.exceptions import ReadOnlyError
from smqtk.representation import get_descriptor_index_impls, \
    get_key_value_store_impls
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import metrics
from smqtk.utils import plugin
from smqtk.utils.bit_utils import bit_vector_to_int_large
from smqtk.utils.bin_utils import report_progress
from smqtk.utils import merge_dict

try:
    from six.moves import cPickle as pickle
except ImportError:
    import pickle

from six.moves import map, zip


class LSHNearestNeighborIndex (NearestNeighborsIndex):
    """
    Locality-sensitive hashing based nearest neighbor index

    This type of algorithm relies on a hashing algorithm to hash descriptors
    such that similar descriptors are hashed the same or similar hash value.
    This allows simpler distance functions (hamming distance) to be performed on
    hashes in order to find nearby bins which are more likely to hold similar
    descriptors.

    LSH nearest neighbor algorithms consist of:
        * Index of descriptors to query over
        * A hashing function that transforms a descriptor vector into a
          hash (bit-vector).
        * Key-Value store of hash values to their set of hashed descriptor
          UUIDs.
        * Nearest neighbor index for indexing bit-vectors (treated as
          descriptors)

    """

    @classmethod
    def is_usable(cls):
        # This "shell" class is always usable, no special dependencies.
        return True

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
        default = super(LSHNearestNeighborIndex, cls).get_default_config()

        lf_default = plugin.make_config(get_lsh_functor_impls())
        default['lsh_functor'] = lf_default

        di_default = plugin.make_config(get_descriptor_index_impls())
        default['descriptor_index'] = di_default

        hi_default = plugin.make_config(get_hash_index_impls())
        default['hash_index'] = hi_default
        default['hash_index_comment'] = "'hash_index' may also be null to " \
                                        "default to a linear index built at " \
                                        "query time."

        h2u_default = plugin.make_config(get_key_value_store_impls())
        default['hash2uuids_kvstore'] = h2u_default

        return default

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: LSHNearestNeighborIndex

        """
        # Controlling merge here so we can control known comment stripping from
        # default config.
        if merge_default:
            merged = cls.get_default_config()
            merge_dict(merged, config_dict)
        else:
            merged = config_dict

        merged['lsh_functor'] = \
            plugin.from_plugin_config(merged['lsh_functor'],
                                      get_lsh_functor_impls())
        merged['descriptor_index'] = \
            plugin.from_plugin_config(merged['descriptor_index'],
                                      get_descriptor_index_impls())

        # Hash index may be None for a default at-query-time linear indexing
        if merged['hash_index'] and merged['hash_index']['type']:
            merged['hash_index'] = \
                plugin.from_plugin_config(merged['hash_index'],
                                          get_hash_index_impls())
        else:
            cls.get_logger().debug("No HashIndex impl given. Passing ``None``.")
            merged['hash_index'] = None

        # remove possible comment added by default generator
        if 'hash_index_comment' in merged:
            del merged['hash_index_comment']

        merged['hash2uuids_kvstore'] = \
            plugin.from_plugin_config(merged['hash2uuids_kvstore'],
                                      get_key_value_store_impls())

        return super(LSHNearestNeighborIndex, cls).from_config(merged, False)

    def __init__(self, lsh_functor, descriptor_index, hash2uuids_kvstore,
                 hash_index=None,
                 distance_method='cosine', read_only=False):
        """
        Initialize LSH algorithm with a hashing functor, descriptor index and
        hash nearest-neighbor index.

        In order to provide out-of-the-box neighbor querying ability, all three
        of the ``descriptor_index``, ``hash_index`` and
        ``hash2uuids_kvstore`` must be provided. The two indices should
        also be fully linked by the mapping provided by the key-value mapping
        provided by the ``hash2uuids_kvstore``. If not, not all descriptors will
        be accessible via the nearest-neighbor query (not referenced in
        ``hash2uuids_kvstore`` map), or the requested number of neighbors might
        not be returned (descriptors hashed in ``hash_index`` disjoint from
        ``descriptor_index``).

        An ``LSHNearestNeighborIndex`` instance is effectively read-only if any
        of its input structures are read-only.

        :param lsh_functor: LSH functor implementation instance.
        :type lsh_functor: smqtk.algorithms.nn_index.lsh.functors.LshFunctor

        :param descriptor_index: Index in which DescriptorElements will be
            stored.
        :type descriptor_index: smqtk.representation.DescriptorIndex

        :param hash2uuids_kvstore: KeyValueStore instance to use for linking a
            hash code, as an integer, in the ``hash_index`` with one or more
            ``DescriptorElement`` instance UUIDs in the given
            ``descriptor_index``.
        :type hash2uuids_kvstore: smqtk.representation.KeyValueStore

        :param hash_index: ``HashIndex`` for indexing unique hash codes using
            hamming distance.

            If this is set to ``None`` (default), we will perform brute-force
            linear neighbor search for each query based on the hash codes
            currently in the hash2uuid index using hamming distance
        :type hash_index: smqtk.algorithms.nn_index.hash_index.HashIndex | None

        :param distance_method: String label of distance method to use for
            determining descriptor similarity (after finding near hashes for a
            given query).

            This must one of the following:
                - "euclidean": Simple euclidean distance between two
                    descriptors (L2 norm).
                - "cosine": Cosine angle distance/similarity between two
                    descriptors.
                - "hik": Histogram intersection distance between two
                    descriptors.
        :type distance_method: str

        :param read_only: If this index should only read from its configured
            descriptor and hash indexes. This will cause a ``ReadOnlyError`` to
            be raised from build_index.
        :type read_only: bool

        :raises ValueError: Invalid distance method specified.

        """
        super(LSHNearestNeighborIndex, self).__init__()

        # TODO(paul.tunison): Add in-memory empty defaults for
        #   descriptor_index/hash2uuids_kvstore attributes.
        self.lsh_functor = lsh_functor
        self.descriptor_index = descriptor_index
        self.hash_index = hash_index
        # Will use with int|long keys and set[collection.Hashable] values.
        self.hash2uuids_kvstore = hash2uuids_kvstore
        self.distance_method = distance_method
        self.read_only = read_only

        self._distance_function = self._get_dist_func(self.distance_method)

    @staticmethod
    def _get_dist_func(distance_method):
        """
        Return appropriate distance function given a string label
        """
        if distance_method == "euclidean":
            return metrics.euclidean_distance
        elif distance_method == "cosine":
            # Inverse of cosine similarity function return
            return metrics.cosine_distance
        elif distance_method == 'hik':
            return metrics.histogram_intersection_distance_fast
        else:
            # TODO: Support scipy/scikit-learn distance methods
            raise ValueError("Invalid distance method label. Must be one of "
                             "['euclidean' | 'cosine' | 'hik']")

    def get_config(self):
        hi_conf = None
        if self.hash_index is not None:
            hi_conf = plugin.to_plugin_config(self.hash_index)
        return {
            "lsh_functor": plugin.to_plugin_config(self.lsh_functor),
            "descriptor_index": plugin.to_plugin_config(self.descriptor_index),
            "hash_index": hi_conf,
            "hash2uuids_kvstore":
                plugin.to_plugin_config(self.hash2uuids_kvstore),
            "distance_method": self.distance_method,
            "read_only": self.read_only,
        }

    def count(self):
        """
        :return: Maximum number of descriptors reference-able via a
            nearest-neighbor query (count of descriptor index). Actual return
            may be smaller of hash2uuids mapping is not complete.
        :rtype: int
        """
        return len(self.descriptor_index)

    def build_index(self, descriptors):
        """
        Build the index over the descriptor data elements. This in turn builds
        the configured hash index if one is set.

        Subsequent calls to this method should rebuild the index, not add to
        it, or raise an exception to as to protect the current index. Rebuilding
        the LSH index involves clearing the set descriptor index, key-value
        store and, if set, the hash index.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        if self.read_only:
            raise ReadOnlyError("Cannot modify container attributes due to "
                                "being in read-only mode.")

        self._log.debug("Clearing and adding new descriptor elements")
        self.descriptor_index.clear()
        self.descriptor_index.add_many_descriptors(descriptors)

        self._log.debug("Generating hash codes")
        state = [0] * 7
        hash_vectors = collections.deque()
        self.hash2uuids_kvstore.clear()
        for d in self.descriptor_index:
            h = self.lsh_functor.get_hash(d.vector())
            hash_vectors.append(h)

            h_int = bit_vector_to_int_large(h)

            # Get, update and reinsert hash UUID set object
            #: :type: set
            hash_uuid_set = self.hash2uuids_kvstore.get(h_int, set())
            hash_uuid_set.add(d.uuid())
            self.hash2uuids_kvstore.add(h_int, hash_uuid_set)

            report_progress(self._log.debug, state, 1.0)
        state[1] -= 1
        report_progress(self._log.debug, state, 0)

        if self.hash_index is not None:
            self._log.debug("Clearing and building hash index of type %s",
                            type(self.hash_index))
            # a build is supposed to clear previous state.
            self.hash_index.build_index(hash_vectors)

    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple
            of the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """
        super(LSHNearestNeighborIndex, self).nn(d, n)

        self._log.debug("generating hash for descriptor")
        d_v = d.vector()
        d_h = self.lsh_functor.get_hash(d_v)

        def comp_descr_dist(d2_v):
            return self._distance_function(d_v, d2_v)

        self._log.debug("getting near hashes")
        hi = self.hash_index
        if hi is None:
            # Make on-the-fly linear index
            hi = LinearHashIndex()
            # not calling ``build_index`` because we already have the int
            # hashes.
            hi.index = numpy.array(list(self.hash2uuids_kvstore.keys()))
        near_hashes, _ = hi.nn(d_h, n)

        self._log.debug("getting UUIDs of descriptors for nearby hashes")
        neighbor_uuids = []
        for h_int in map(bit_vector_to_int_large, near_hashes):
            # If descriptor hash not in our map, we effectively skip it.
            # Get set of descriptor UUIDs for a hash code.
            #: :type: set[collections.Hashable]
            near_uuids = self.hash2uuids_kvstore.get(h_int, set())
            # Accumulate matching descriptor UUIDs to a list.
            neighbor_uuids.extend(near_uuids)
        self._log.debug("-- matched %d UUIDs", len(neighbor_uuids))

        self._log.debug("getting descriptors for neighbor_uuids")
        neighbors = \
            list(self.descriptor_index.get_many_descriptors(neighbor_uuids))

        self._log.debug("ordering descriptors via distance method '%s'",
                        self.distance_method)
        self._log.debug('-- getting element vectors')
        neighbor_vectors = elements_to_matrix(neighbors,
                                              report_interval=1.0)
        self._log.debug('-- calculating distances')
        distances = list(map(comp_descr_dist, neighbor_vectors))
        self._log.debug('-- ordering')
        ordered = sorted(zip(neighbors, distances),
                         key=lambda p: p[1])
        self._log.debug('-- slicing top n=%d', n)
        return list(zip(*(ordered[:n])))


# Marking only LSH as the valid impl, otherwise the hash index default would
#   also be picked up (because it also descends from NearestNeighborsIndex).
NN_INDEX_CLASS = LSHNearestNeighborIndex
