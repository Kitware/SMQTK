"""
Future home of IQR LSH implementation based on Chapel Hill paper / sample code.
"""
__author__ = 'purg'

import numpy
import numpy.matlib

from smqtk.similarity_index import (
    SimilarityIndex,
)


class ITQSimilarityIndex (SimilarityIndex):
    """
    Nearest neighbor implementation using Iterative Quantization (ITQ)
    """

    @classmethod
    def is_usable(cls):
        # Internal implementation, so no dependencies besides things like numpy,
        # which if we don't have, nothing will work.
        return True

    def __init__(self, bit_length, itq_iterations=50):
        """
        Initialize ITQ similarity index instance (not the index itself).

        :param bit_length: Number of bits used to represent descriptors (hash
            code). This must be greater than 0.
        :type bit_length: int

        :param itq_iterations: Number of iterations for the ITQ algorithm to
            perform. This must be greater than 0.
        :type itq_iterations: int

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
        #: :type: numpy.core.multiarray.ndarray | None
        self._mean_vector = None

        # Index descriptor elements we are indexing. This will be index aligned
        # with the proceeding _C matrix.
        self._descr_cache = []

        # Principar component vector that will get multiplied against new
        # vectors
        #: :type: numpy.core.multiarray.ndarray | None
        self._pc_vec = None

        # NxB matrix, where N is the number of descriptors indexed, and B is the
        # configured bit length.
        # TODO: This should probably be, instead, some index by small-code
        #: :type: numpy.core.multiarray.ndarray | None
        self._C = None

        # BxB rotation matrix found by ITQ
        #: :type: numpy.core.multiarray.ndarray | None
        self._R = None

        # DxB transformation matrix (?), where D is descriptor dimensionality,
        # and B is configured bit length.
        #: :type: numpy.core.multiarray.ndarray | None
        self._transform = None

    @staticmethod
    def _find_itq_rotation(v, n_iter):
        """
        Finds a rotation of the PCA embedded data. Number of iterations must be
        greater than 0.

        :param v: 2D numpy array, n*c PCA embedded data, n is the number of images
            and c is the code length
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
        r = numpy.random.randn(bit,bit)
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

        # ! Bug in original implementation from UNC-CH, wasn't using final
        # ! rotation matrix in return ``b`` calculation.
        # # Make B binary
        # b = ux
        # b[b < 0] = 0
        # Make B binary matrix using final rotation matrix
        z = numpy.dot(v, r)
        b = numpy.zeros(z.shape)
        b[z >= 0] = 1

        return b, r

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return self._C.shape[0] if self._C else 0

    def build_index(self, descriptors):
        """
        Build the index over the descriptor data elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """
        # TODO: Sub-sample down descriptors to use for PCA + ITQ

        x = []
        self._descr_cache = []
        for d in descriptors:
            self._descr_cache.append(d)
            x.append(d.vector())
        if not x:
            raise ValueError("No descriptors given!")
        x = numpy.array(x)

        # center the data, VERY IMPORTANT for ITQ to work
        self._mean_vector = numpy.mean(x, axis=0)
        x = x - numpy.matlib.repmat(self._mean_vector, x.shape[0], 1)

        # PCA
        # TODO: I think this section is incorrect compared to UNC-CH MATLAB code
        c = numpy.cov(x.transpose())
        pc, l, v2 = numpy.linalg.svd(c)
        self._pc_vec = pc[:, :self._bit_len]
        xx = numpy.dot(x, self._pc_vec)

        # ITQ to find optimal rotation.
        #   `c` is the output codes for matrix `x`
        #   `r` is the rotation found by ITQ
        self._C, self._R = self._find_itq_rotation(xx, self._itq_iter_num)
        self._C = self._C.astype(int)

        # TODO: Create binary tree with descriptors as leaves?


    def save_index(self):
        # TODO
        raise NotImplementedError()

    def load_index(self):
        # TODO
        raise NotImplementedError()

    def nn(self, d, n=1):
        sc = self.get_small_code(d)

        # Return descriptors and distances vectors
        r = []
        d = []
        # Terminate when we have reached the number of matches requested, or
        # we've pulled everything in our index.
        while len(r) < n or len(r) < self.count():
            pass

        return r[:n], d[:n]

    def get_small_code(self, descriptor):
        """
        Generate the small-code for the given descriptor.

        This only works if we have an index loaded.

        :param descriptor: Descriptor to generate the small code for.
        :type descriptor: smqtk.data_rep.DescriptorElement

        :return: Compacted N-bit small-code.
        :rtype: numpy.core.multiarray.ndarray

        """
        v1 = descriptor.vector() - self._mean_vector
        v2 = numpy.dot(v1, self._pc_vec)
        z = numpy.dot(v2, self._R)
        b = numpy.zeros(z.shape, dtype=int)
        b[z >= 0] = 1
        return b
