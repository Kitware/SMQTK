"""
Future home of IQR LSH implementation based on Chapel Hill paper / sample code.
"""
__author__ = 'purg'

import numpy
import numpy.matlib

from smqtk.similarity_index import (
    SimilarityIndex,
)
from smqtk.utils import bit_utils, SimpleTimer
from smqtk.utils import distance_functions


class ITQSimilarityIndex (SimilarityIndex):
    """
    Nearest neighbor implementation using Iterative Quantization (ITQ)

    Currently uses cosine distance as a fine-grain distance metric when
    computing neighbors.

    IMPORTANT:
        For consistency, we treat bit vectors such that the bit at index 0 is
        considered the most significant bit.

    """

    @classmethod
    def is_usable(cls):
        # Internal implementation, so no dependencies besides things like numpy,
        # which if we don't have, nothing will work.
        return True

    def __init__(self, bit_length, itq_iterations=50, distance_method='cosine'):
        """
        Initialize ITQ similarity index instance (not the index itself).

        :param bit_length: Number of bits used to represent descriptors (hash
            code). This must be greater than 0.
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

        """
        assert bit_length > 0, "Must be given a bit length greater than 1 " \
                               "(one)!"
        assert itq_iterations > 0, "Must be given a number of iterations " \
                                   "greater than 1 (one)!"

        # Number of bits we convert descriptors into
        self._bit_len = int(bit_length)
        # Number of iterations ITQ performs
        self._itq_iter_num = itq_iterations

        # Vector of mean feature values. Centers "train" set, used to "center"
        # additional descriptors when computing nearest neighbors.
        #: :type: numpy.core.multiarray.ndarray
        self._mean_vector = None

        # Index descriptor elements we are indexing. This will be index aligned
        # with the proceeding _c matrix.
        #   - This is probably OK to keep along side the ``_code_table`` since
        #       they'll just be sharing references, not copies, of the
        #       descriptor elements.
        #: :type: list[smqtk.data_rep.DescriptorElement]
        self._descr_cache = []

        # rotation matrix of shape [d, b], found by ITQ process, to use to
        # transform new descriptors into binary hash decision vector.
        #: :type: numpy.core.multiarray.ndarray[float] | None
        self._r = None

        # Hash table mapping small-codes to a list of DescriptorElements mapped
        # by that code
        #: :type: dict[int, list[smqtk.data_rep.DescriptorElement]]
        self._code_table = {}

        if distance_method == "euclidean":
            #: :type: (ndarray, ndarray) -> ndarray
            self._dist_func = distance_functions.euclidean_distance
        elif distance_method == "cosine":
            # Inverse of cosine similarity function return
            #: :type: (ndarray, ndarray) -> ndarray
            self._dist_func = \
                lambda i, j: 1.0 - distance_functions.cosine_similarity(i, j)
        elif distance_method == 'hik':
            #: :type: (ndarray, ndarray) -> ndarray
            self._dist_func = distance_functions.histogram_intersection_distance
        else:
            raise ValueError("Invalid distance method label. Must be one of "
                             "['euclidean' | 'cosine' | 'hik']")

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self._descr_cache)

    @staticmethod
    def _find_itq_rotation(v, n_iter):
        """
        Finds a rotation of the PCA embedded data. Number of iterations must be
        greater than 0.

        This is equivalent to the ITQ function from UNC-CH's implementation.

        :param v: 2D numpy array, n*c PCA embedded data, n is the number of data
            elements and c is the code length.
        :type v: numpy.core.multiarray.ndarray

        :param n_iter: max number of iterations, 50 is usually enough
        :type n_iter: int

        :return: [b, r]
           b: 2D numpy array, n*c binary matrix,
           r: 2D numpy array, the c*c rotation matrix found by ITQ
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        #initialize with an orthogonal random rotation
        bit = v.shape[1]
        r = numpy.random.randn(bit, bit)
        u11, s2, v2 = numpy.linalg.svd(r)
        r = u11[:, :bit]

        #ITQ to find optimal rotation
        for _ in range(n_iter):
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

        Subsequent calls to this method should rebuild the index, not add to it.

        The first part of this method is equivalent to the compressITQ function
        from UNC-CH's implementation.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """
        # TODO: Sub-sample down descriptors to use for PCA + ITQ
        #       - Harry was also working on an iterative training approach so
        #           that we only have to have a limited number of vectors in
        #           memory at a time.

        with SimpleTimer("Creating descriptor matrix", self._log.info):
            x = []
            self._descr_cache = []
            for d in descriptors:
                self._descr_cache.append(d)
                x.append(d.vector())
            if not x:
                raise ValueError("No descriptors given!")
            x = numpy.array(x)

        with SimpleTimer("Centering data", self._log.info):
            # center the data, VERY IMPORTANT for ITQ to work
            self._mean_vector = numpy.mean(x, axis=0)
            x = x - numpy.matlib.repmat(self._mean_vector, x.shape[0], 1)

        # PCA
        # TODO: I think this section is incorrect compared to UNC-CH MATLAB code
        with SimpleTimer("Computing PCA transformation", self._log.info):
            c = numpy.cov(x.transpose())
            pc, l, v2 = numpy.linalg.svd(c)
            pc_vec = pc[:, :self._bit_len]
            # adjust input data given PC vector
            xx = numpy.dot(x, pc_vec)

            # c = numpy.cov(x)
            # pc, l = numpy.linalg.eig(c)
            # pc_vec = pc[:self._bit_len]
            # xx = x * pc_vec

        # ITQ to find optimal rotation.
        #   `c` is the output codes for matrix `x`
        #   `r` is the rotation found by ITQ
        with SimpleTimer("Performing ITQ to find optimal rotation",
                         self._log.info):
            c, self._r = self._find_itq_rotation(xx, self._itq_iter_num)
            # De-adjust rotation with PC vector
            self._r = numpy.dot(pc_vec, self._r)

        # Converting bit-vectors proved faster than creating new codes over
        # again (~0.01s vs ~0.04s for size 80 vectors).
        with SimpleTimer("Converting bitvectors into small codes",
                         self._log.info):
            for code_vec, descr in zip(c, self._descr_cache):
                packed = bit_utils.bit_vector_to_int(code_vec)
                self._code_table.setdefault(packed, []).append(descr)

        pass

    def save_index(self):
        # TODO
        raise NotImplementedError()

    def load_index(self):
        # TODO
        raise NotImplementedError()

    def get_small_code(self, descriptor):
        """
        Generate the small-code for the given descriptor.

        This only works if we have an index loaded.

        :param descriptor: Descriptor to generate the small code for.
        :type descriptor: smqtk.data_rep.DescriptorElement

        :return: The descriptor's vector and the compacted N-bit small-code as
            an integer.
        :rtype: int

        """
        v = descriptor.vector()
        z = numpy.dot(v - self._mean_vector, self._r)
        b = numpy.zeros(z.shape, dtype=numpy.uint8)
        b[z >= 0] = 1
        return v, bit_utils.bit_vector_to_int(b)

    def _neighbor_codes(self, c, d):
        """
        Iterate through small-codes of length ``b``, where ``b`` is the number
        of bits this index is configured for, that are ``d`` hamming distance
        away from query code ``c``.

        This will yield a number of elements equal to ``nCr(b, d)``.

        We expect ``d`` to be the integer hamming distance,
        e.g. h(001101, 100101) == 2, not 0.333.

        :param c: Query small-code integer
        :type c: int

        :param d: Integer hamming distance
        :type d: int

        """
        if not d:
            yield c
            raise StopIteration()

        for fltr in bit_utils.iter_perms(self._bit_len, d):
            yield c ^ fltr

    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.data_rep.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.
        :rtype: (tuple[smqtk.data_rep.DescriptorElement], tuple[float])

        """
        d_vec, d_sc = self.get_small_code(d)

        # Process:
        #   Collect codes/descriptors from incrementally more distance bins
        #   until we have at least the number of neighbors requested. Then,
        #   compute fine-grain distances with those against query descriptor to
        #   get final order and return distance values.
        #: :type: list[smqtk.data_rep.DescriptorElement]
        neighbors = []
        termination_count = min(n, self.count())
        h_dist = 0
        while len(neighbors) < termination_count and h_dist <= self._bit_len:
            # Get codes of hamming dist, ``h_dist``, from ``d``'s code
            codes = self._neighbor_codes(d_sc, h_dist)
            for c in codes:
                neighbors.extend(self._code_table.get(c, []))
            h_dist += 1
        neighbors = neighbors[:n]

        # Compute fine-grain distance measurements for collected elements + sort
        distances = []
        for d_elem in neighbors:
            distances.append(self._dist_func(d_vec, d_elem.vector()))

        t = zip(neighbors, distances)
        for i, (d_elem, dist) in enumerate(sorted(zip(neighbors, distances),
                                                  key=lambda e: e[1])):
            neighbors[i] = d_elem
            distances[i] = dist

        return neighbors, distances
