import logging
import os.path

import numpy

from smqtk.algorithms.nn_index.lsh.functors import LshFunctor
from smqtk.representation.descriptor_element import elements_to_matrix


__author__ = "paul.tunison@kitware.com"


class ItqFunctor (LshFunctor):
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

    IMPORTANT:
        For consistency, we treat bit vectors such that the bit at index 0 is
        considered the most significant bit ()bit-endian).

    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, mean_vec_filepath=None, rotation_filepath=None,
                 bit_length=8, itq_iterations=50, random_seed=None):
        """
        Initialize IQR functor.

        File Path Note
        --------------
        Model filepaths should point to numpy files and include the ".npy" file
        suffix. If not, then when files are saves, numpy will automatically
        append the ".npy" suffix (e.g. "example" -> "example.npy"). This can
        lead to not being able to use the same configuration for fitting and
        application, as the file name's as given will not be found.

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

        :param bit_length: Number of bits used to represent descriptors (hash
            code). This must be greater than 0. If given an existing
        :type bit_length: int

        :param itq_iterations: Number of iterations for the ITQ algorithm to
            perform. This must be greater than 0.
        :type itq_iterations: int

        :param random_seed: Integer to use as the random number generator seed.
        :type random_seed: int

        """
        super(ItqFunctor, self).__init__()

        self.mean_vec_filepath = mean_vec_filepath
        self.rotation_filepath = rotation_filepath
        self.bit_length = bit_length
        self.itq_iterations = itq_iterations
        self.random_seed = random_seed

        # Model components
        self.mean_vec = None
        self.rotation = None

        self.load_model()

    def get_config(self):
        return {
            "mean_vec_filepath": self.mean_vec_filepath,
            "rotation_filepath": self.rotation_filepath,
            "bit_length": self.bit_length,
            "itq_iterations": self.itq_iterations,
            "random_seed": self.random_seed,
        }

    def has_model(self):
        return (self.mean_vec is not None) and (self.rotation is not None)

    def load_model(self):
        if (self.mean_vec_filepath and
                os.path.isfile(self.mean_vec_filepath) and
                self.rotation_filepath and
                os.path.isfile(self.rotation_filepath)):
            self.mean_vec = numpy.load(self.mean_vec_filepath)
            self.rotation = numpy.load(self.rotation_filepath)

    def save_model(self):
        if (self.mean_vec_filepath and self.rotation_filepath and
                self.mean_vec is not None and self.rotation is not None):
            numpy.save(self.mean_vec_filepath, self.mean_vec)
            numpy.save(self.rotation_filepath, self.rotation)

    def _find_itq_rotation(self, v, n_iter):
        """
        Finds a rotation of the PCA embedded data. Number of iterations must be
        greater than 0.

        This is equivalent to the ITQ function from UNC-CH's implementation.

        ``self`` is used only for logging. Otherwise this has no side effects
        on this instance.

        :param v: 2D numpy array, n*c PCA embedded data, n is the number of
            data elements and c is the code length.
        :type v: numpy.core.multiarray.ndarray

        :param n_iter: max number of iterations, 50 is usually enough
        :type n_iter: int

        :return: [b, r]
           b: 2D numpy array, n*c binary matrix,
           r: 2D numpy array, the c*c rotation matrix found by ITQ
        :rtype: numpy.core.multiarray.ndarray[bool],
            numpy.core.multiarray.ndarray[float]

        """
        # initialize with an orthogonal random rotation
        bit = v.shape[1]
        if self.random_seed is not None:
            numpy.random.seed(self.random_seed)
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
        # TODO: Could move this step up one level and just return rotation mat?
        z = numpy.dot(v, r)
        b = numpy.zeros(z.shape, dtype=numpy.bool)
        b[z >= 0] = True

        return b, r

    def fit(self, descriptors):
        """
        Fit the ITQ model given the input set of descriptors

        :param descriptors: Iterable of ``DescriptorElement`` vectors to fit
            the model to.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :raises RuntimeError: There is already a model loaded

        :return: Matrix hash codes for provided descriptors in order.
        :rtype: numpy.ndarray[bool]

        """
        if self.has_model():
            raise RuntimeError("Model components have already been loaded.")

        self._log.info("Creating matrix of descriptors for fitting")
        dbg_report_interval = None
        if self.logger().getEffectiveLevel() <= logging.DEBUG:
            dbg_report_interval = 1.0  # seconds
        x = elements_to_matrix(list(descriptors),
                               report_interval=dbg_report_interval)
        self._log.debug("descriptor matrix shape: %s", x.shape)

        self._log.info("Centering data")
        self.mean_vec = numpy.mean(x, axis=0)
        x -= self.mean_vec

        self._log.info("Computing PCA transformation")
        # numpy and matlab observation format is flipped, thus the added
        # transpose.
        self._log.debug("-- computing covariance")
        c = numpy.cov(x.transpose())

        # Direct translation from UNC matlab code
        # - eigen vectors are the columns of ``pc``
        self._log.debug('-- computing linalg.eig')
        l, pc = numpy.linalg.eig(c)
        # ordered by greatest eigenvalue magnitude, keeping top ``bit_len``
        self._log.debug('-- computing top pairs')
        top_pairs = sorted(zip(l, pc.transpose()),
                           key=lambda p: p[0],
                           reverse=1
                           )[:self.bit_length]

        # # Harry translation -- Uses singular values / vectors, not eigen
        # # - singular vectors are the rows of pc
        # pc, l, _ = numpy.linalg.svd(c)
        # top_pairs = sorted(zip(l, pc),
        #                    key=lambda p: p[0],
        #                    reverse=1
        #                    )[:self.bit_length]

        # Eigen-vectors of top ``bit_len`` magnitude eigenvalues
        self._log.debug("-- top vector extraction")
        pc_top = numpy.array([p[1] for p in top_pairs]).transpose()
        self._log.debug("-- transform centered data by PC matrix")
        xx = numpy.dot(x, pc_top)

        self._log.info("Performing ITQ to find optimal rotation")
        c, self.rotation = self._find_itq_rotation(xx, self.itq_iterations)
        # De-adjust rotation with PC vector
        self.rotation = numpy.dot(pc_top, self.rotation)

        self.save_model()

        return c

    def get_hash(self, descriptor):
        """
        Get the locality-sensitive hash code for the input descriptor.

        :param descriptor: Descriptor vector we should generate the hash of.
        :type descriptor: numpy.ndarray[float]

        :return: Generated bit-vector as a numpy array of booleans.
        :rtype: numpy.ndarray[bool]

        """
        z = numpy.dot(descriptor - self.mean_vec, self.rotation)
        b = numpy.zeros(z.shape, dtype=bool)
        b[z >= 0] = True
        return b
