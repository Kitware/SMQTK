"""
Home of IQR LSH implementation based on UNC Chapel Hill paper / sample code.
"""

import heapq
import os.path as osp

import numpy
import numpy.matlib

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.representation.code_index import get_code_index_impls
from smqtk.representation.code_index.memory import MemoryCodeIndex
from smqtk.utils import (
    bit_utils,
    distance_functions,
    file_utils,
    plugin,
    SimpleTimer,
)

__author__ = "paul.tunison@kitware.com"


class ITQNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest neighbor implementation using Iterative Quantization (ITQ), a method
    to convert a descriptor (e.g. 4000-dim vector) to few bits (e.g. 64 bits).

    The method first appeared in
    ```
    Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A Procrustes
    Approach to Learning Binary Codes. In CVPR 2011.
    ```
    It was originally implemented in Matlab by Yunchao Gong
    (yunchao@cs.unc.edu).

    It may be the case that there the given index instance is not empty. In this
    case, there is an existing computed index, but no internal state yet for
    this ITQ index.

    IMPORTANT:
        For consistency, we treat bit vectors such that the bit at index 0 is
        considered the most significant bit.

    """

    @classmethod
    def is_usable(cls):
        # Internal implementation, so no dependencies besides things like numpy,
        # which if we don't have, nothing will work.
        return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as arguments,
        turning those argument names into configuration dictionary keys. If any
        of those arguments have defaults, we will add those values into the
        configuration dictionary appropriately. The dictionary returned should
        only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        default = super(ITQNearestNeighborsIndex, cls).get_default_config()

        # replace ``code_index`` with nested plugin configuration
        index_conf = plugin.make_config(get_code_index_impls)
        if default['code_index'] is not None:
            # Only overwrite default config if there is a default value
            index_conf.update(plugin.to_plugin_config(default['code_index']))
        default['code_index'] = index_conf

        return default

    @classmethod
    def from_config(cls, config_dict):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary.

        This implementation nests the configuration of the CodeIndex
        implementation to use. If there is a ``code_index`` in the configuration
        dictionary, it should be a nested plugin specification dictionary, as
        specified by the ``smqtk.utils.plugin.from_config`` method.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :return: ITQ similarity index instance
        :rtype: ITQNearestNeighborsIndex

        """
        # Transform nested plugin stuff into actual classes.
        config_dict['code_index'] = \
            plugin.from_plugin_config(config_dict['code_index'],
                                      get_code_index_impls)

        return super(ITQNearestNeighborsIndex, cls).from_config(config_dict)

    def __init__(self, mean_vec_filepath=None,
                 rotation_filepath=None,
                 code_index=MemoryCodeIndex(),
                 # Index building parameters
                 bit_length=8, itq_iterations=50, distance_method='cosine',
                 random_seed=None):
        """
        Initialize ITQ similarity index instance.

        This implementation allows optional persistant storage of a built model
        via providing file paths for the ``mean_vec_filepath`` and
        ``rotation_filepath`` parameters.


        The Code Index
        --------------
        ``code_index`` should be an instance of a CodeIndex implementation
        class.

        The ``build_index`` call will clear the provided index anything in the
        index provided. For safety, make sure to check the index provided so as
        to not accidentally erase data.

        When providing existing mean_vector and rotation matrix caches, the
        ``code_index`` may be populated with codes. Pre-populated entries in the
        provided code index should have been generated from the same rotation
        and mean vector models provided, else nearest-neighbor query performance
        will not be as desired.

        A more advanced use case includes providing a code index that is
        update-able in the background. This is valid, assuming there is
        proper locking mechanisms in the code index.


        Build parameters
        ----------------
        Parameters after file path parameters are only related to building the
        index. When providing existing mean, rotation and code elements, these
        can be safely ignored.


        :raise ValueError: Invalid argument values.

        :param mean_vec_filepath: Optional file location to load/store the mean
            vector when initialized and/or built. When None, this will only be
            stored in memory. This will use numpy to save/load, so this should
            have a ``.npy`` suffix, or one will be added at save time.
        :type mean_vec_filepath: str

        :param rotation_filepath: Optional file location to load/store the
            rotation matrix when initialize and/or built. When None, this will
            only be stored in memory. This will use numpy to save/load, so this
            should have a ``.npy`` suffix, or one will be added at save time.
        :type rotation_filepath: str

        :param code_index: CodeIndex instance to use.
        :type code_index: smqtk.representation.code_index.CodeIndex

        :param bit_length: Number of bits used to represent descriptors (hash
            code). This must be greater than 0. If given an existing
        :type bit_length: int

        :param itq_iterations: Number of iterations for the ITQ algorithm to
            perform. This must be greater than 0.
        :type itq_iterations: int

        :param distance_method: String label of distance method to use. This
            must one of the following:
                - "euclidean": Simple euclidean distance between two descriptors
                    (L2 norm).
                - "cosine": Cosine angle distance/similarity between two
                    descriptors.
                - "hik": Histogram intersection distance between two
                    descriptors.
        :type distance_method: str

        :param random_seed: Integer to use as the random number generator seed.
        :type random_seed: int

        """
        self._mean_vec_cache_filepath = mean_vec_filepath
        self._rotation_cache_filepath = rotation_filepath

        # maps small-codes to a list of DescriptorElements mapped by that code
        self._code_index = code_index

        # Number of bits we convert descriptors into
        self._bit_len = int(bit_length)
        # Number of iterations ITQ performs
        self._itq_iter_num = int(itq_iterations)
        # Optional fixed random seed
        self._rand_seed = None if random_seed is None else int(random_seed)

        assert bit_length > 0, "Must be given a bit length greater than 1 " \
                               "(one)!"
        assert itq_iterations > 0, "Must be given a number of iterations " \
                                   "greater than 1 (one)!"

        # Vector of mean feature values. Center of "train" set, and used to
        # "center" additional descriptors when computing small codes.
        #: :type: numpy.core.multiarray.ndarray[float]
        self._mean_vector = None
        if self._mean_vec_cache_filepath and \
                osp.isfile(self._mean_vec_cache_filepath):
            self._log.debug("Loading existing descriptor vector mean")
            #: :type: numpy.core.multiarray.ndarray[float]
            self._mean_vector = numpy.load(self._mean_vec_cache_filepath)

        # rotation matrix of shape [d, b], found by ITQ process, to use to
        # transform new descriptors into binary hash decision vector.
        #: :type: numpy.core.multiarray.ndarray[float]
        self._r = None
        if self._rotation_cache_filepath and \
                osp.isfile(self._rotation_cache_filepath):
            self._log.debug("Loading existing descriptor rotation matrix")
            #: :type: numpy.core.multiarray.ndarray[float]
            self._r = numpy.load(self._rotation_cache_filepath)

        self._dist_method = distance_method
        self._dist_func = self._get_dist_func(distance_method)

    def get_config(self):
        return {
            "mean_vec_filepath": self._mean_vec_cache_filepath,
            "rotation_filepath": self._rotation_cache_filepath,
            "code_index": plugin.to_plugin_config(self._code_index),
            "bit_length": self._bit_len,
            "itq_iterations": self._itq_iter_num,
            "distance_method": self._dist_method,
            'random_seed': self._rand_seed,
        }

    @staticmethod
    def _get_dist_func(distance_method):
        """
        Return appropriate distance function given a string label
        """
        if distance_method == "euclidean":
            #: :type: (ndarray, ndarray) -> ndarray
            return distance_functions.euclidean_distance
        elif distance_method == "cosine":
            # Inverse of cosine similarity function return
            #: :type: (ndarray, ndarray) -> ndarray
            return lambda i, j: 1.0 - distance_functions.cosine_similarity(i, j)
        elif distance_method == 'hik':
            #: :type: (ndarray, ndarray) -> ndarray
            return distance_functions.histogram_intersection_distance_fast
        else:
            raise ValueError("Invalid distance method label. Must be one of "
                             "['euclidean' | 'cosine' | 'hik']")

    def count(self):
        """
        :return: Number of descriptor elements in this index.
        :rtype: int
        """
        if self._code_index:
            return self._code_index.count()
        else:
            return 0

    def _find_itq_rotation(self, v, n_iter):
        """
        Finds a rotation of the PCA embedded data. Number of iterations must be
        greater than 0.

        This is equivalent to the ITQ function from UNC-CH's implementation.

        ``self`` is used only for logging. Otherwise this has no side effects on
        this instance.

        :param v: 2D numpy array, n*c PCA embedded data, n is the number of data
            elements and c is the code length.
        :type v: numpy.core.multiarray.ndarray

        :param n_iter: max number of iterations, 50 is usually enough
        :type n_iter: int

        :return: [b, r]
           b: 2D numpy array, n*c binary matrix,
           r: 2D numpy array, the c*c rotation matrix found by ITQ
        :rtype: numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray

        """
        # initialize with an orthogonal random rotation
        bit = v.shape[1]
        r = numpy.random.randn(bit, bit)
        u11, s2, v2 = numpy.linalg.svd(r)
        r = u11[:, :bit]

        # ITQ to find optimal rotation
        self._log.debug("ITQ iterations to determine optimal rotation: %d",
                        n_iter)
        for i in range(n_iter):
            self._log.debug("ITQ iter %d", i + 1)
            z = numpy.dot(v, r)
            ux = numpy.ones(z.shape) * (-1)
            ux[z >= 0] = 1
            c = numpy.dot(ux.transpose(), v)
            ub, sigma, ua = numpy.linalg.svd(c)
            r = numpy.dot(ua, ub.transpose())

        # Make B binary matrix using final rotation matrix
        #   - original code returned B transformed by second to last rotation
        #       matrix, there by returning, de-synchronized matrices
        #   - Recomputing Z here so as to generate up-to-date B for the final
        #       rotation matrix computed.
        # TODO: Could move this step up one level and just return rotation mat
        z = numpy.dot(v, r)
        b = numpy.zeros(z.shape, dtype=numpy.uint8)
        b[z >= 0] = True

        return b, r

    def build_index(self, descriptors):
        """
        Build the index over the descriptor data elements.

        The first part of this method is equivalent to the compressITQ function
        from UNC-CH's implementation.

        :raises RuntimeError: A current data model is loaded, or the current
            CodeIndex is not empty.
        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        # Halt if we are going to overwrite a loaded mean/rotation cache.
        if not (self._mean_vector is None and self._r is None):
            raise RuntimeError("Current ITQ model is not empty (cached mean / "
                               "rotation). For the sake of protecting data, we "
                               "are not proceeding.")
        # Halt if the code index currently isn't empty
        if self.count():
            raise RuntimeError("Current CodeIndex instance is not empty. For "
                               "the sake of protecting data, we are not "
                               "proceeding.")

        self._log.debug("Using %d length bit-vectors", self._bit_len)

        # TODO: Sub-sample down descriptors to use for PCA + ITQ
        #       - Harry was also working on an iterative training approach so
        #           that we only have to have a limited number of vectors in
        #           memory at a time.
        if self._rand_seed:
            numpy.random.seed(self._rand_seed)

        with SimpleTimer("Creating descriptor matrix", self._log.info):
            x = []
            #: :type: list[smqtk.representation.DescriptorElement]
            descr_cache = []
            for d in descriptors:
                descr_cache.append(d)
                x.append(d.vector())
            if not x:
                raise ValueError("No descriptors given!")
            x = numpy.array(x)

        with SimpleTimer("Centering data", self._log.info):
            # center the data, VERY IMPORTANT for ITQ to work
            self._mean_vector = numpy.mean(x, axis=0)
            x -= self._mean_vector
        if self._mean_vec_cache_filepath:
            with SimpleTimer("Saving mean vector", self._log.info):
                file_utils.safe_create_dir(osp.dirname(self._mean_vec_cache_filepath))
                numpy.save(self._mean_vec_cache_filepath, self._mean_vector)

        # PCA
        with SimpleTimer("Computing PCA transformation", self._log.info):
            # numpy and matlab observation format is flipped, thus added
            # transpose
            self._log.debug("-- computing covariance")
            c = numpy.cov(x.transpose())

            # Direct translation
            # - eigen vectors are the columns of ``pc``
            self._log.debug('-- computing linalg.eig')
            l, pc = numpy.linalg.eig(c)
            # ordered by greatest eigenvalue magnitude, keeping top ``bit_len``
            self._log.debug('-- computing top pairs')
            top_pairs = sorted(zip(l, pc.transpose()),
                               key=lambda p: p[0],
                               reverse=1
                               )[:self._bit_len]

            # # Harry translation -- Uses singular values / vectors, not eigen
            # # - singular vectors are the rows of pc
            # pc, l, _ = numpy.linalg.svd(c)
            # top_pairs = sorted(zip(l, pc),
            #                    key=lambda p: p[0],
            #                    reverse=1
            #                    )[:self._bit_len]

            # Eigen-vectors of top ``bit_len`` magnitude eigenvalues
            self._log.debug("-- top vector extraction")
            pc_top = numpy.array([p[1] for p in top_pairs]).transpose()
            self._log.debug("-- transform centered data by PC matrix")
            xx = numpy.dot(x, pc_top)

        # ITQ to find optimal rotation.
        #   `c` is the output codes for matrix `x`
        #   `r` is the rotation found by ITQ
        with SimpleTimer("Performing ITQ to find optimal rotation",
                         self._log.info):
            c, self._r = self._find_itq_rotation(xx, self._itq_iter_num)
            # De-adjust rotation with PC vector
            self._r = numpy.dot(pc_top, self._r)
        if self._rotation_cache_filepath:
            with SimpleTimer("Saving rotation matrix", self._log.info):
                file_utils.safe_create_dir(osp.dirname(self._rotation_cache_filepath))
                numpy.save(self._rotation_cache_filepath, self._r)

        # Populating small-code index
        #   - Converting bit-vectors proved faster than creating new codes over
        #       again (~0.01s vs ~0.04s for 80 vectors).
        with SimpleTimer("Clearing code index", self._log.info):
            self._code_index.clear()
        with SimpleTimer("Converting bit-vectors into small codes, inserting "
                         "into code index", self._log.info):
            self._code_index.add_many_descriptors(
                (bit_utils.bit_vector_to_int(c[i]), descr_cache[i])
                for i in xrange(c.shape[0])
            )
        # NOTE: If a sub-sampling effect is implemented above, this will have to
        #       change to querying for descriptor vectors individually since the
        #       ``c`` matrix will not encode all descriptors at that point. This
        #       will be slower unless we think of something else. Could probably
        #       map the small code generation function by bringing it outside of
        #       the class.

    def get_small_code(self, descriptor):
        """
        Generate the small-code for the given descriptor.

        This only works if we have an index loaded, meaning we have a rotation
        matrix.

        :param descriptor: Descriptor to generate the small code for.
        :type descriptor: smqtk.representation.DescriptorElement

        :return: The descriptor's vector, the n-bit vector, and the compacted
            N-bit small-code as an integer.
        :rtype: numpy.core.multiarray.ndarray[float],
                numpy.core.multiarray.ndarray[numpy.uint8],
                int

        """
        v = descriptor.vector()
        z = numpy.dot(v - self._mean_vector, self._r)
        b = numpy.zeros(z.shape, dtype=numpy.uint8)
        b[z >= 0] = 1
        return v, b, bit_utils.bit_vector_to_int(b)

    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """
        d_vec, _, d_sc = self.get_small_code(d)

        # Extract the `n` nearest codes to the code of the query descriptor
        # - a code may associate with multiple hits, but its a safe assumption
        #   that if we get the top `n` codes, which exist because there is at
        #   least one element in association with it,
        code_set = self._code_index.codes()
        # TODO: Optimize this step
        #: :type: list[int]
        near_codes = \
            heapq.nsmallest(n, code_set,
                            lambda e:
                                distance_functions.hamming_distance(d_sc, e)
                            )

        # Collect descriptors from subsequently farther away bins until we have
        # >= `n` descriptors, which we will more finely sort after this.
        #: :type: list[smqtk.representation.DescriptorElement]
        neighbors = []
        termination_count = min(n, self.count())
        for nc in near_codes:
            neighbors.extend(self._code_index.get_descriptors(nc))
            # Break out if we've collected >= `n` descriptors, as descriptors
            # from more distance codes are likely to not be any closer.
            if len(neighbors) >= termination_count:
                break

        # Compute fine-grain distance measurements for collected elements + sort
        # for d_elem in neighbors:
        #     distances.append(self._dist_func(d_vec, d_elem.vector()))
        def comp_neighbor_dist(neighbor):
            return self._dist_func(d_vec, neighbor.vector())
        distances = map(comp_neighbor_dist, neighbors)

        # Sort by distance, return top n
        ordered = sorted(zip(neighbors, distances), key=lambda p: p[1])
        neighbors, distances = zip(*(ordered[:n]))
        return neighbors, distances
