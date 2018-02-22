"""
References/Resources:
- UNC paper: http://www.cs.unc.edu/~lazebnik/publications/cvpr11_small_code.pdf
- ACM reference: http://dl.acm.org/citation.cfm?id=2191779
- GitHub with matlab implementation:
  https://github.com/willard-yuan/hashing-baseline-for-image-retrieval
"""
from collections import Sequence
from copy import deepcopy
import logging

import numpy

from smqtk.algorithms.nn_index.lsh.functors import LshFunctor
from smqtk.representation import get_data_element_impls
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import merge_dict, plugin
from smqtk.utils.bin_utils import report_progress

from six import BytesIO


class ItqFunctor (LshFunctor):
    """
    LSH hash code functor implementation using Iterative Quantization (ITQ), a
    method to convert a descriptor (e.g. 4000-dim vector) to a fewer number of
    bits (e.g. 64 bits).

    The method first appeared in
    ```
    Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A Procrustes
    Approach to Learning Binary Codes. In CVPR 2011.
    ```
    It was originally implemented in Matlab by Yunchao Gong
    (yunchao@cs.unc.edu).

    IMPORTANT:
        For consistency, we treat bit vectors such that the bit at index 0 is
        considered the most significant bit (big-endian).

    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        default = super(ItqFunctor, cls).get_default_config()

        # Cache element parameters need to be split out into sub-configurations
        data_element_default_config = \
            plugin.make_config(get_data_element_impls())
        default['mean_vec_cache'] = data_element_default_config
        # Need to deepcopy source to prevent modifications on one sub-config
        # from reflecting in the other.
        default['rotation_cache'] = deepcopy(data_element_default_config)

        return default

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the JSON-compliant
        configuration dictionary encapsulating initialization arguments.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: ItqFunctor

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        data_element_impls = get_data_element_impls()
        # Mean vector cache element.
        mean_vec_cache = None
        if config_dict['mean_vec_cache'] and \
                config_dict['mean_vec_cache']['type']:
            mean_vec_cache = plugin.from_plugin_config(
                config_dict['mean_vec_cache'], data_element_impls)
        config_dict['mean_vec_cache'] = mean_vec_cache
        # Rotation matrix cache element.
        rotation_cache = None
        if config_dict['rotation_cache'] and \
                config_dict['rotation_cache']['type']:
            rotation_cache = plugin.from_plugin_config(
                config_dict['rotation_cache'], data_element_impls)
        config_dict['rotation_cache'] = rotation_cache

        return super(ItqFunctor, cls).from_config(config_dict, False)

    def __init__(self, mean_vec_cache=None, rotation_cache=None,
                 bit_length=8, itq_iterations=50, normalize=None,
                 random_seed=None):
        """
        Initialize IQR functor.

        File Path Note
        --------------
        Model filepaths should point to numpy files and include the ".npy" file
        suffix. If not, then when files are saves, numpy will automatically
        append the ".npy" suffix (e.g. "example" -> "example.npy"). This can
        lead to not being able to use the same configuration for fitting and
        application, as the file name's as given will not be found.

        :param mean_vec_cache: Optional data element to load/store the mean
            vector when initialized and/or built. When None, this will only be
            stored in memory.
        :type mean_vec_cache: smqtk.representation.DataElement

        :param rotation_cache: Optional data element to load/store the rotation
            matrix when initialize and/or built. When None, this will only be
            stored in memory.
        :type rotation_cache: smqtk.representation.DataElement

        :param bit_length: Number of bits used to represent descriptors (hash
            code). This must be greater than 0. If given an existing
        :type bit_length: int

        :param itq_iterations: Number of iterations for the ITQ algorithm to
            perform. This must be greater than 0.
        :tyepe itq_iterations: int

        :param normalize: Normalize input vectors when fitting and generation
            hash vectors using ``numpy.linalg.norm``. This may either
            be ``None``, disabling normalization, or any valid value that
            could be passed to the ``ord`` parameter in ``numpy.linalg.norm``
            for 1D arrays. This is ``None`` by default (no normalization).

            Normalization affects the value of the mean vector and rotation
            matrix. This means that model products produced reflect the
            normalization value use when training and the same normalization
            value, like the bit_length value, must be used when loading cached
            models again for later use.
        :type normalize: None | int | float | str

        :param random_seed: Integer to use as the random number generator seed.
        :type random_seed: int

        """
        super(ItqFunctor, self).__init__()

        self.mean_vec_cache_elem = mean_vec_cache
        self.rotation_cache_elem = rotation_cache
        self.bit_length = bit_length
        self.itq_iterations = itq_iterations
        self.normalize = normalize
        self.random_seed = random_seed

        # Validate normalization parameter by trying it on a random vector
        if normalize is not None:
            self._norm_vector(numpy.random.rand(8))

        # Model components
        self.mean_vec = None
        self.rotation = None

        self.load_model()

    def _norm_vector(self, v):
        """
        Class standard array normalization. Normalized along max dimension (a=0
        for a 1D array, a=1 for a 2D array, etc.).

        If ``self.normalize`` is None, ``v`` is returned without modification.

        :param v: Vector to normalize
        :type v: numpy.ndarray

        :return: Returns the normalized version of input array ``v``.
        :rtype: numpy.ndarray

        """
        if self.normalize is not None:
            n = numpy.linalg.norm(v, self.normalize, v.ndim - 1,
                                  keepdims=True)
            # replace 0's with 1's, preventing div-by-zero
            n[n == 0.] = 1.
            return v / n

        # Normalization off
        return v

    def get_config(self):
        # If no cache elements (set to None), return default plugin configs.
        c = merge_dict(self.get_default_config(), {
            "bit_length": self.bit_length,
            "itq_iterations": self.itq_iterations,
            "normalize": self.normalize,
            "random_seed": self.random_seed,
        })
        if self.mean_vec_cache_elem:
            c['mean_vec_cache'] = \
                plugin.to_plugin_config(self.mean_vec_cache_elem)
        if self.rotation_cache_elem:
            c['rotation_cache'] = \
                plugin.to_plugin_config(self.rotation_cache_elem)
        return c

    def has_model(self):
        return (self.mean_vec is not None) and (self.rotation is not None)

    def load_model(self):
        if (self.mean_vec_cache_elem
                and not self.mean_vec_cache_elem.is_empty()
                and self.rotation_cache_elem
                and not self.rotation_cache_elem.is_empty()):
            self.mean_vec = \
                numpy.load(BytesIO(self.mean_vec_cache_elem.get_bytes()))
            self.rotation = \
                numpy.load(BytesIO(self.rotation_cache_elem.get_bytes()))

    def save_model(self):
        # Check that we have cache elements set, they are writable and that we
        # have something to save to them.
        if (self.mean_vec_cache_elem and self.rotation_cache_elem and
                self.mean_vec_cache_elem.writable() and
                self.rotation_cache_elem.writable() and
                self.mean_vec is not None and self.rotation is not None):
            b = BytesIO()
            numpy.save(b, self.mean_vec)
            self.mean_vec_cache_elem.set_bytes(b.getvalue())

            b = BytesIO()
            numpy.save(b, self.rotation)
            self.rotation_cache_elem.set_bytes(b.getvalue())

    def _find_itq_rotation(self, v, n_iter):
        """
        Finds a rotation of the PCA embedded data. Number of iterations must be
        greater than 0.

        This is equivalent to the ITQ function from UNC-CH's implementation.

        ``self`` is used only for logging. Otherwise this has no side effects
        on this instance.

        :param v: 2D numpy array, [n, c] shape PCA embedded data, ``n`` is the
            number of data elements and ``c`` is the code length.
        :type v: numpy.core.multiarray.ndarray

        :param n_iter: max number of iterations, 50 is usually enough
        :type n_iter: int

        :return: [b, r]
           b: 2D numpy array, [n, c] shape binary matrix,
           r: 2D numpy array, the [c, c] shape rotation matrix found by ITQ
        :rtype: numpy.core.multiarray.ndarray[bool],
            numpy.core.multiarray.ndarray[float]

        """
        # Pull num bits from PCA-projected descriptors
        bit = v.shape[1]
        # initialize with an orthogonal random rotation
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
            # TODO: @numba.jit decorate
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

    def fit(self, descriptors, use_multiprocessing=True):
        """
        Fit the ITQ model given the input set of descriptors.

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

        dbg_report_interval = None
        if self.get_logger().getEffectiveLevel() <= logging.DEBUG:
            dbg_report_interval = 1.0  # seconds
        if not isinstance(descriptors, Sequence):
            self._log.info("Creating sequence from iterable")
            descriptors_l = []
            rs = [0]*7
            for d in descriptors:
                descriptors_l.append(d)
                report_progress(self._log.debug, rs, dbg_report_interval)
            descriptors = descriptors_l
        if len(descriptors[0].vector()) < self.bit_length:
            raise ValueError("Input descriptors have fewer features than "
                             "requested bit encoding. Hash codes will be "
                             "smaller than requested due to PCA decomposition "
                             "result being bound by number of features.")

        self._log.info("Creating matrix of descriptors for fitting")
        x = elements_to_matrix(descriptors, report_interval=dbg_report_interval,
                               use_multiprocessing=use_multiprocessing)
        self._log.debug("descriptor matrix shape: %s", x.shape)

        self._log.debug("Info normalizing descriptors by factor: %s",
                        self.normalize)
        x = self._norm_vector(x)

        self._log.info("Centering data")
        self.mean_vec = numpy.mean(x, axis=0)
        x -= self.mean_vec

        self._log.info("Computing PCA transformation")
        self._log.debug("-- computing covariance")
        # ``cov`` wants each row to be a feature and each column an observation
        # of those features. Thus, each column should be a descriptor vector,
        # thus we need the transpose here.
        c = numpy.cov(x.transpose())

        if True:
            # Direct translation from UNC matlab code
            # - eigen vectors are the columns of ``pc``
            self._log.debug('-- computing linalg.eig')
            l, pc = numpy.linalg.eig(c)
            self._log.debug('-- ordering eigen vectors by descending eigen '
                            'value')
        else:
            # Harry translation -- Uses singular values / vectors, not eigen
            # - singular vectors are the columns of pc
            self._log.debug('-- computing linalg.svd')
            pc, l, _ = numpy.linalg.svd(c)
            self._log.debug('-- ordering singular vectors by descending '
                            'singular value')

        # Same ordering method for both eig/svd sources.
        l_pc_ordered = sorted(zip(l, pc.transpose()), key=lambda p: p[0],
                              reverse=1)

        self._log.debug("-- top vector extraction")
        # Only keep the top ``bit_length`` vectors after ordering by descending
        # value magnitude.
        # - Transposing vectors back to column-vectors.
        pc_top = numpy.array([p[1] for p in l_pc_ordered[:self.bit_length]])\
            .transpose()
        self._log.debug("-- project centered data by PC matrix")
        v = numpy.dot(x, pc_top)

        self._log.info("Performing ITQ to find optimal rotation")
        c, self.rotation = self._find_itq_rotation(v, self.itq_iterations)
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
        z = numpy.dot(self._norm_vector(descriptor) - self.mean_vec,
                      self.rotation)
        b = numpy.zeros(z.shape, dtype=bool)
        b[z >= 0] = True
        return b
