from __future__ import division
import logging

import numpy as np

from smqtk.algorithms.nn_index.lsh.functors import LshFunctor
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils.bin_utils import report_progress


class SimpleRPFunctor (LshFunctor):
    '''
    This class is meant purely as a baseline comparison for other
    LshFunctors and NNIndex plugins. It is not meant to be used in
    production, as it is unlikely to produce a quality index.
    '''

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, bit_length=8, normalize=None, random_seed=None):
        super(SimpleRPFunctor, self).__init__()

        self.bit_length = bit_length
        self.normalize = normalize
        self.random_seed = random_seed

        # Model components
        self.mean_vec = None

    def _norm_vector(self, v):
        """
        Class standard array normalization. Normalized along max dimension
        (a=0 for a 1D array, a=1 for a 2D array, etc.).

        :param v: Vector to normalize
        :type v: numpy.ndarray

        :return: Returns the normalized version of input array ``v``.
        :rtype: numpy.ndarray

        """
        vm = v - self.mean_vec
        if self.normalize is None:
            # Normalization off
            return vm

        n = np.linalg.norm(vm, ord=self.normalize, axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            return np.nan_to_num(vm/n)

    def get_config(self):
        return {
            "bit_length": self.bit_length,
            "normalize": self.normalize,
            "random_seed": self.random_seed,
        }

    def has_model(self):
        return (self.mean_vec is not None)

    def fit(self, descriptors, use_multiprocessing=True):
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

        dbg_report_interval = None
        if self.logger().getEffectiveLevel() <= logging.DEBUG:
            dbg_report_interval = 1.0  # seconds
        if not hasattr(descriptors, "__len__"):
            self._log.info("Creating sequence from iterable")
            descriptors_l = []
            rs = [0]*7
            for d in descriptors:
                descriptors_l.append(d)
                report_progress(self._log.debug, rs, dbg_report_interval)
            descriptors = descriptors_l
        self._log.info("Creating matrix of descriptors for fitting")
        x = elements_to_matrix(
            descriptors, report_interval=dbg_report_interval,
            use_multiprocessing=use_multiprocessing)
        self._log.debug("descriptor matrix shape: %s", x.shape)
        n, dim = x.shape

        self._log.debug("Generating random projections")
        np.random.seed(self.random_seed)
        self.rps = np.random.randn(dim, self.bit_length)

        self._log.debug("Info normalizing descriptors with norm type: %s",
                        self.normalize)
        return self.get_hash(x)

    def get_hash(self, descriptor):
        b = (self._norm_vector(descriptor).dot(self.rps) >= 0.0)
        return b.squeeze()
