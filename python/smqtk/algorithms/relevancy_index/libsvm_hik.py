import copy
import cPickle
import os.path as osp

import numpy

from smqtk.algorithms.relevancy_index import RelevancyIndex
from smqtk.utils.distance_kernel import (
    compute_distance_matrix
)
from smqtk.utils.metrics import histogram_intersection_distance
from smqtk.utils.parallel import parallel_map

try:
    import svm
    import svmutil
except ImportError:
    svm = None
    svmutil = None


__author__ = "paul.tunison@kitware.com"


class LibSvmHikRelevancyIndex (RelevancyIndex):
    """
    Uses libSVM python interface, using histogram intersection, to implement
    IQR ranking.
    """

    # Dictionary of parameter/value pairs that will be passed to libSVM during
    # the model trail phase. Parameters that are flags, i.e. have no values,
    # should be given an empty string ('') value.
    SVM_TRAIN_PARAMS = {
        '-q': '',
        '-t': 5,  # Specified the use of HI kernel, 5 is unique to custom build
        '-b': 1,
        '-c': 2,
        '-g': 0.0078125,
    }

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.

        Required valid presence of svm and svmutil modules

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        return svm and svmutil

    def __init__(self, descr_cache_filepath=None, autoneg_select_ratio=1,
                 multiprocess_fetch=False, cores=None):
        """
        Initialize a new or existing index.

        TODO ::
        - input optional known background descriptors, i.e. descriptors for
            things that would otherwise always be considered a negative example.

        :param descr_cache_filepath: Optional path to store/load descriptors
            we index.
        :type descr_cache_filepath: None | str

        :param autoneg_select_ratio: Number of maximally distant descriptors to
            select from our descriptor cache for each positive example provided
            when no negative examples are provided for ranking.

            This must be an integer. Default value of 1. It is advisable not to
            make this value too large.
        :type autoneg_select_ratio: int

        :param multiprocess_fetch: Use multiprocessing vs threading when
            fetching descriptor vectors during ``build_index``. Default is
            False.
        :type multiprocess_fetch: bool

        :param cores: Cores to use when performing parallel operations. A value
            of None means to use all available cores.
        :type cores: int | None

        """
        super(LibSvmHikRelevancyIndex, self).__init__()

        self.descr_cache_fp = descr_cache_filepath
        self.autoneg_select_ratio = int(autoneg_select_ratio)
        self.multiprocess_fetch = multiprocess_fetch
        self.cores = cores

        # Descriptor elements in this index
        self._descr_cache = []
        # Local serialization of descriptor vectors. Used when for computing
        # distances of SVM support vectors for Platt Scaling
        self._descr_matrix = None
        # Mapping of descriptor vectors to their index in the cache, and
        # subsequently in the distance kernel
        self._descr2index = {}
        # # Distance kernel matrix (symmetric)
        # self._dist_kernel = None

        if self.descr_cache_fp and osp.exists(self.descr_cache_fp):
            with open(self.descr_cache_fp, 'rb') as f:
                descriptors = cPickle.load(f)
                # Temporarily unsetting so we don't cause an extra write inside
                # build_index.
                self.descr_cache_fp = None
                self.build_index(descriptors)
                self.descr_cache_fp = descr_cache_filepath

    @staticmethod
    def _gen_w1_weight(num_pos, num_neg):
        """
        Return w1 weight parameter based on pos and neg exemplars
        """
        return max(1.0, num_neg/float(num_pos))

    @classmethod
    def _gen_svm_parameter_string(cls, num_pos, num_neg):
        params = copy.copy(cls.SVM_TRAIN_PARAMS)
        params['-w1'] = cls._gen_w1_weight(num_pos, num_neg)
        return ' '.join((' '.join((str(k), str(v))) for k, v in params.items()))

    def get_config(self):
        return {
            "descr_cache_filepath": self.descr_cache_fp,
            'autoneg_select_ratio': self.autoneg_select_ratio,
            'multiprocess_fetch': self.multiprocess_fetch,
            'cores': self.cores,
        }

    def count(self):
        return len(self._descr_cache)

    def build_index(self, descriptors):
        """
        Build the index based on the given iterable of descriptor elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors: collections.Iterable[smqtk.representation.DescriptorElement]

        """
        # ordered cache of descriptors in our index.
        self._descr_cache = []
        # Reverse mapping of a descriptor's vector to its index in the cache
        # and subsequently in the distance kernel.
        self._descr2index = {}
        # matrix for creating distance kernel
        self._descr_matrix = []

        def get_vector(d):
            return d, d.vector()

        vector_iter = parallel_map(get_vector, descriptors,
                                   name='vector_iter',
                                   use_multiprocessing=self.multiprocess_fetch,
                                   cores=self.cores,
                                   ordered=True)

        for i, (d, v) in enumerate(vector_iter):
            self._descr_cache.append(d)
            self._descr_matrix.append(v)
            self._descr2index[tuple(v)] = i
        self._descr_matrix = numpy.array(self._descr_matrix)

        # TODO: (?) For when we optimize SVM SV kernel computation
        # self._dist_kernel = \
        #    compute_distance_kernel(self._descr_matrix,
        #                            histogram_intersection_distance2,
        #                            row_wise=True)

        if self.descr_cache_fp:
            with open(self.descr_cache_fp, 'wb') as f:
                cPickle.dump(self._descr_cache, f, -1)

    def rank(self, pos, neg):
        """
        Rank the currently indexed elements given ``pos`` positive and ``neg``
        negative exemplar descriptor elements.

        :param pos: Iterable of positive exemplar DescriptorElement instances.
            This may be optional for some implementations.
        :type pos: collections.Iterable[smqtk.representation.DescriptorElement]

        :param neg: Iterable of negative exemplar DescriptorElement instances.
            This may be optional for some implementations.
        :type neg: collections.Iterable[smqtk.representation.DescriptorElement]

        :return: Map of indexed descriptor elements to a rank value between
            [0, 1] (inclusive) range, where a 1.0 means most relevant and 0.0
            meaning least relevant.
        :rtype: dict[smqtk.representation.DescriptorElement, float]

        """
        # Notes:
        # - Pos and neg exemplars may be in our index.

        #
        # SVM model training
        #
        # Copy pos descriptors into a set for repeated iteration
        #: :type: set[smqtk.representation.DescriptorElement]
        pos = set(pos)
        # Creating training matrix and labels
        train_labels = []
        train_vectors = []
        num_pos = 0
        for d in pos:
            train_labels.append(+1)
            train_vectors.append(d.vector().tolist())
            num_pos += 1
        self._log.debug("Positives given: %d", num_pos)

        # When no negative examples are given, naively pick most distant example
        # in our dataset, using HI metric, for each positive example
        neg_autoselect = set()
        if not neg:
            self._log.info("Auto-selecting negative examples. (%d per positive)",
                           self.autoneg_select_ratio)
            # ``train_vectors`` only composed of positive examples at this point
            for p in pos:
                # where d is the distance vector to descriptor elements in cache
                d = histogram_intersection_distance(p.vector(),
                                                    self._descr_matrix)
                # Scan vector for max distance index
                # - Allow variable number of maximally distance descriptors to
                #   be picked per positive.
                m_set = {}  # track most distance neighbors
                m_val = -float('inf')  # track smallest distance of most distant neighbors
                for i in xrange(d.size):
                    if d[i] > m_val:
                        m_set[d[i]] = i
                        if len(m_set) > self.autoneg_select_ratio:
                            if m_val in m_set:
                                del m_set[m_val]
                        m_val = min(m_set)
                for i in m_set.itervalues():
                    neg_autoselect.add(self._descr_cache[i])
            # Remove any positive examples from auto-selected results
            neg_autoselect.difference_update(pos)
            self._log.debug("Auto-selected negative descriptors [%d]: %s",
                            len(neg_autoselect), neg_autoselect)

        num_neg = 0
        for d in neg:
            train_labels.append(-1)
            train_vectors.append(d.vector().tolist())
            num_neg += 1
        for d in neg_autoselect:
            train_labels.append(-1)
            train_vectors.append(d.vector().tolist())
            num_neg += 1

        if not num_pos:
            raise ValueError("No positive examples provided.")
        elif not num_neg:
            raise ValueError("No negative examples provided.")

        # Training SVM model
        self._log.debug("online model training")
        svm_problem = svm.svm_problem(train_labels, train_vectors)
        svm_model = svmutil.svm_train(svm_problem,
                                      self._gen_svm_parameter_string(num_pos,
                                                                     num_neg))
        if svm_model.l == 0:
            raise RuntimeError("SVM Model learning failed")

        #
        # Platt Scaling for probability rankings
        #

        self._log.debug("making test distance matrix")
        # Number of support vectors
        # Q: is this always the same as ``svm_model.l``?
        num_SVs = sum(svm_model.nSV[:svm_model.nr_class])
        # Support vector dimensionality
        dim_SVs = len(train_vectors[0])
        # initialize matrix they're going into
        svm_SVs = numpy.ndarray((num_SVs, dim_SVs), dtype=float)
        for i, nlist in enumerate(svm_model.SV[:svm_SVs.shape[0]]):
            svm_SVs[i, :] = [n.value for n in nlist[:len(train_vectors[0])]]
        # compute matrix of distances from support vectors to index elements
        # TODO: Optimize this step by caching SV distance vectors
        #       - It is known that SVs are vectors from the training data, so
        #           if the same descriptors are given to this function
        #           repeatedly (which is the case for IQR), this can be faster
        #           because we're only computing at most a few more distance
        #           vectors against our indexed descriptor matrix, and the rest
        #           have already been computed before.
        #       - At worst, we're effectively doing this call because each SV
        #           needs to have its distance vector computed.
        svm_test_k = compute_distance_matrix(svm_SVs, self._descr_matrix,
                                             histogram_intersection_distance,
                                             row_wise=True)

        self._log.debug("Platt scaling")
        # the actual platt scaling stuff
        weights = numpy.array(svm_model.get_sv_coef()).flatten()
        margins = numpy.dot(weights, svm_test_k)
        rho = svm_model.rho[0]
        probA = svm_model.probA[0]
        probB = svm_model.probB[0]
        #: :type: numpy.core.multiarray.ndarray
        probs = 1.0 / (1.0 + numpy.exp((margins - rho) * probA + probB))

        # Detect whether we need to flip probabilities
        # - Probability of input positive examples should have a high
        #   probability score among the generated probabilities of our index.
        # - If the positive example probabilities show to be in the lower 50%,
        #   flip the generated probabilities, since its experimentally known
        #   that the SVM will change which index it uses to represent a
        #   particular class label occasionally, which influences the Platt
        #   scaling apparently.
        pos_vectors = numpy.array(train_vectors[:num_pos])
        pos_test_k = compute_distance_matrix(svm_SVs, pos_vectors,
                                             histogram_intersection_distance,
                                             row_wise=True)
        pos_margins = numpy.dot(weights, pos_test_k)
        #: :type: numpy.core.multiarray.ndarray
        pos_probs = 1.0 / (1.0 + numpy.exp((pos_margins - rho) * probA + probB))
        # Check if average positive probability is less than the average index
        # probability. If so, the platt scaling probably needs to be flipped.
        if (pos_probs.sum() / pos_probs.size) < (probs.sum() / probs.size):
            self._log.debug("inverting probabilities")
            probs = 1. - probs

        rank_pool = dict(zip(self._descr_cache, probs))
        return rank_pool


RELEVANCY_INDEX_CLASS = LibSvmHikRelevancyIndex
