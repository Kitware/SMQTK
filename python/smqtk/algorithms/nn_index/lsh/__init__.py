"""
This module contains a general base locality-sensitive-hashing algorithm for
nearest neighbor indexing, and various implementations of LSH functors for use
in the base.
"""
import cPickle
import os
import time

import numpy

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.algorithms.nn_index.hash_index import get_hash_index_impls
from smqtk.algorithms.nn_index.hash_index.linear import LinearHashIndex
from smqtk.algorithms.nn_index.lsh.functors import get_lsh_functor_impls
from smqtk.representation import get_descriptor_index_impls
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import distance_functions
from smqtk.utils import plugin
from smqtk.utils.bit_utils import bit_vector_to_int_large
from smqtk.utils.errors import ReadOnlyError


__author__ = "paul.tunison@kitware.com"


class LSHNearestNeighborIndex (NearestNeighborsIndex):
    """
    Locality-sensitive hashing based nearest neighbor index

    This type of algorithm relies on a hashing algorithm to hash descriptors
    such that similar descriptors are hashed the same or similarly. This allows
    simpler distance functions to be performed on hashes in order to find
    nearby bins which are more likely to hold similar descriptors.

    LSH nearest neighbor algorithms consist of:
        * Index of descriptors to query over
        * A hashing function that transforms a descriptor vector into a
          hash (bit-vector).
        * Nearest neighbor index for indexing bit-vectors (treated as
          descriptors)

    """

    @classmethod
    def is_usable(cls):
        # This shell class is always usable
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

        lf_default = plugin.make_config(get_lsh_functor_impls)
        default['lsh_functor'] = lf_default

        di_default = plugin.make_config(get_descriptor_index_impls)
        default['descriptor_index'] = di_default

        hi_default = plugin.make_config(get_hash_index_impls)
        default['hash_index'] = hi_default

        return default

    @classmethod
    def from_config(cls, config_dict):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :return: Constructed instance from the provided config.
        :rtype: LSHNearestNeighborIndex

        """
        merged = cls.get_default_config()
        merged.update(config_dict)

        merged['lsh_functor'] = \
            plugin.from_plugin_config(merged['lsh_functor'],
                                      get_lsh_functor_impls)
        merged['descriptor_index'] = \
            plugin.from_plugin_config(merged['descriptor_index'],
                                      get_descriptor_index_impls)
        merged['hash_index'] = \
            plugin.from_plugin_config(merged['hash_index'],
                                      get_hash_index_impls)

        return super(LSHNearestNeighborIndex, cls).from_config(merged)

    def __init__(self, lsh_functor, descriptor_index, hash_index=None,
                 hash2uuid_cache_filepath=None,
                 distance_method='cosine', read_only=False):
        """
        Initialize LSH algorithm with a hashing functor, descriptor index and
        hash nearest-neighbor index.

        In order to provide out-of-the-box neighbor querying ability, all three
        of the ``descriptor_index``, ``hash_index`` and
        ``hash2uuid_cache_filepath`` must be provided. The two indices should
        also be fully linked by the mapping provided by the
        ``hash2uuid_cache_filepath``. If not, not all descriptors will be
        accessible via the neighbor query (not referenced in map), or the
        requested number of neighbors might not be returned (indexed hashes
        don't reference descriptors in the descriptor index).

        :param lsh_functor: LSH functor implementation instance.
        :type lsh_functor: smqtk.algorithms.nn_index.lsh.functors.LshFunctor

        :param descriptor_index: Index in which DescriptorElements will be
            stored.
        :type descriptor_index: smqtk.representation.DescriptorIndex

        :param hash_index: ``HashIndex`` for indexing unique hash codes using
            hamming distance.

            If this is set to ``None`` (default), we will perform brute-force
            linear neighbor search for each query based on the hash codes
            currently in the hash2uuid index using hamming distance
        :type hash_index: smqtk.algorithms.nn_index.hash_index.HashIndex | None

        :param hash2uuid_cache_filepath: Path to save the hash code to
            descriptor UUID mapping. If provided, this is written to when
            ``build_index`` is called.

            If not provided, a call to ``build_index`` is required in order to
            build the mapping, which is then not saved.
        :type hash2uuid_cache_filepath: str

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
            be raised from build_index. The exception to this is if the

        :raises ValueError: Invalid distance method specified.

        """
        super(LSHNearestNeighborIndex, self).__init__()

        self.lsh_functor = lsh_functor
        self.descriptor_index = descriptor_index
        self.hash_index = hash_index
        self.hash2uuid_cache_filepath = hash2uuid_cache_filepath
        self.distance_method = distance_method
        self.read_only = read_only

        #: :type: dict[int|long, set[collections.Hashable]]
        self._hash2uuid = {}
        self._distance_function = self._get_dist_func(self.distance_method)

        # Load hash2uuid model if it exists
        if self.hash2uuid_cache_filepath and \
                os.path.isfile(self.hash2uuid_cache_filepath):
            with open(self.hash2uuid_cache_filepath) as f:
                #: :type: dict[int|long, set[collections.Hashable]]
                self._hash2uuid = cPickle.load(f)

    @staticmethod
    def _get_dist_func(distance_method):
        """
        Return appropriate distance function given a string label
        """
        if distance_method == "euclidean":
            return distance_functions.euclidean_distance
        elif distance_method == "cosine":
            # Inverse of cosine similarity function return
            return distance_functions.cosine_distance
        elif distance_method == 'hik':
            return distance_functions.histogram_intersection_distance_fast
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
            "hash2uuid_cache_filepath": self.hash2uuid_cache_filepath,
            "distance_method": self.distance_method,
            "read_only": self.read_only,
        }

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self.descriptor_index)

    def build_index(self, descriptors):
        """
        Build the index over the descriptor data elements.

        Subsequent calls to this method should rebuild the index, not add to
        it, or raise an exception to as to protect the current index.

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
        self._hash2uuid = \
            self.build_from_descriptor_index(self.descriptor_index,
                                             self.hash_index,
                                             self.lsh_functor)

        if self.hash2uuid_cache_filepath:
            self._log.debug("Writing out hash2uuid map: %s",
                            self.hash2uuid_cache_filepath)
            with open(self.hash2uuid_cache_filepath, 'w') as f:
                cPickle.dump(self._hash2uuid, f)

    @classmethod
    def build_from_descriptor_index(cls, descriptor_index, hash_index,
                                    hash_functor):
        """
        Use this method to build the hash index and hash-UUID mapping from
        an existing ``DescriptorIndex`` and hash functor.

        We return the hash-to-UUID mapping which should be saved to file via
        ``pickle`` in order to be provided in future
        ``LSHNearestNeighborIndex`` configurations.

        :raises ValueError: If there is nothing in the provided
            ``descriptor_index``. The ``hash_index`` will not be modified
            (it actually raises the exception).

        :param descriptor_index: Existing ``DescriptorIndex`` to build from.
        :type descriptor_index: smqtk.representation.DescriptorIndex

        :param hash_index: ``HashIndex`` to build with generated hash codes.
        :type hash_index: smqtk.algorithms.nn_index.hash_index.HashIndex

        :param hash_functor: ``LshFunctor`` to generate hash codes with.
        :type hash_functor: smqtk.algorithms.nn_index.lsh.functors.LshFunctor

        :return: Hash to ``DescriptorElement`` UUID mapping.
        :rtype: dict[int, collections.Hashable]

        """
        #: :type: dict[int|long, set[collections.Hashable]]
        hash2uuid = {}

        def iter_add_hashes():
            """
            Helper to generate hash codes for descriptors as well as add to map
            """
            l = s = time.time()
            for d in descriptor_index.iterdescriptors():
                h = hash_functor.get_hash(d.vector())
                h_int = bit_vector_to_int_large(h)
                if h_int not in hash2uuid:
                    yield h
                    hash2uuid[h_int] = set()

                    t = time.time()
                    if t - l >= 1.0:
                        n = len(hash2uuid)
                        cls.logger().debug("yielding %f hashes per second "
                                           "(%d of %d total)",
                                           n / (t - s), n,
                                           descriptor_index.count())
                        l = t

                hash2uuid[h_int].add(d.uuid())

        if hash_index is None:
            # Scan through above function to fill in hash2uuid mapping
            list(iter_add_hashes())
        else:
            cls.logger().debug("Building hash index from unique hash codes")
            hash_index.build_index(iter_add_hashes())

        return hash2uuid

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

        self._log.debug("generating has for descriptor")
        d_v = d.vector()
        d_h = self.lsh_functor.get_hash(d_v)

        def comp_descr_dist(d2_v):
            return self._distance_function(d_v, d2_v)

        self._log.debug("getting near hashes")
        hi = self.hash_index
        # Make on-the-fly linear index if we weren't originally set with one
        if hi is None:
            hi = LinearHashIndex()
            # not calling ``build_index`` because we already have the int
            # hashes.
            hi.index = numpy.array(self._hash2uuid.keys())
        hashes, hash_dists = hi.nn(d_h, n)

        self._log.debug("getting UUIDs of descriptors for hashes")
        neighbor_uuids = []
        for h_int in map(bit_vector_to_int_large, hashes):
            neighbor_uuids.extend(self._hash2uuid.get(h_int, ()))
        self._log.debug("-- matched %d UUIDs", len(neighbor_uuids))

        self._log.debug("getting descriptors for neighbor_uuids")
        neighbors = \
            list(self.descriptor_index.get_many_descriptors(*neighbor_uuids))

        self._log.debug("ordering descriptors via distance method '%s'",
                        self.distance_method)
        self._log.debug('-- getting element vectors')
        neighbor_vectors = elements_to_matrix(neighbors,
                                              report_interval=1.0)
        self._log.debug('-- calculating distances')
        distances = map(comp_descr_dist, neighbor_vectors)
        self._log.debug('-- ordering')
        ordered = sorted(zip(neighbors, distances),
                         key=lambda p: p[1])
        self._log.debug('-- slicing top n=%d', n)
        return zip(*(ordered[:n]))


# Marking only LSH as the valid impl, otherwise the hash index default would
#   also be picked up (because it also descends from NearestNeighborsIndex).
NN_INDEX_CLASS = LSHNearestNeighborIndex
