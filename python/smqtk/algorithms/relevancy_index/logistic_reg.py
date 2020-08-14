import collections
import os.path as osp

import numpy
import six
from six.moves import range, zip
from six.moves import cPickle as pickle

from smqtk.algorithms.relevancy_index import RelevancyIndex
from smqtk.utils.parallel import parallel_map

try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None
    sklearn = None


class LogisticRegRelevancyIndex (RelevancyIndex):
    """
    Uses Logistic regression python interface, using cosine distance,
    to implement IQR ranking.
    """

    LR_TRAIN_PARAMS = {
        "penalty": "l2",
        "dual": True,  
        "class_weight": "balanced",
        "random_state": 2,
        "multi_class": "ovr",
        "warm_start": False,
        "n_jobs": 1,
    }

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        return LogisticRegression and sklearn

    def __init__(self, autoneg_select_ratio=1,
                 multiprocess_fetch=False, cores=None):
        """
        Initialize a new or existing index.
        TODO ::
        - input optional known background descriptors, i.e. descriptors for
            things that would otherwise always be considered a negative example.
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
        super(LogisticRegRelevancyIndex, self).__init__()

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

    @classmethod
    def _gen_lr_parameter_string(cls):
        return cls.LR_TRAIN_PARAMS
    
    def get_config(self):
        return {
            'autoneg_select_ratio': self.autoneg_select_ratio,
            'multiprocess_fetch': self.multiprocess_fetch,
            'cores': self.cores,
        }
    
    def __len__(self):
        return self.count()
    
    def count(self):
        return len(self._descr_cache)

    def build_index(self, descriptors):
        """
        Build the index based on the given iterable of descriptor elements.
        Subsequent calls to this method should rebuild the index, not add to
        it.
        :raises ValueError: No data available in the given iterable.
        :param descriptors:
            Iterable of descriptor elements to build index over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]
        """
        # ordered cache of descriptors in our index.
        self._descr_cache = []
        # Reverse mapping of a descriptor's vector to its index in the cache
        # and subsequently in the distance kernel.
        self._descr2index = {}
        # matrix for creating distance kernel
        self._descr_matrix = []

        def get_vector(d_elem):
            return d_elem, d_elem.vector()

        # noinspection PyTypeChecker
        vector_iter = parallel_map(get_vector, descriptors,
                                   name='vector_iter',
                                   use_multiprocessing=self.multiprocess_fetch,
                                   cores=self.cores,
                                   ordered=True)

        for i, (d, v) in enumerate(vector_iter):
            self._descr_cache.append(d)
            # ``_descr_matrix`` is a list, currently.
            # noinspection PyUnresolvedReferences
            self._descr_matrix.append(v)
            self._descr2index[tuple(v)] = i
        self._descr_matrix = numpy.array(self._descr_matrix)
        
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

        # When no negative examples are given, naively pick most distant
        # example in our dataset, using HI metric, for each positive example
        neg_autoselect = set()
        # Copy neg descriptors into a set for testing size.
        if not isinstance(neg, collections.Sized):
            #: :type: set[smqtk.representation.DescriptorElement]
            neg = set(neg)
        if not neg:
            self._log.info("Auto-selecting negative examples. (%d per "
                           "positive)", self.autoneg_select_ratio)
            # ``train_vectors`` only composed of positive examples at this
            # point.
            for p in pos:
                # Where d is the distance vector to descriptor elements in
                # cache.
                d = sklearn.metrics.pairwise.cosine_distances(p.vector().reshape(1, -1),
                                                    self._descr_matrix)[0,:]

                # Scan vector for max distance index
                # - Allow variable number of maximally distance descriptors to
                #   be picked per positive.
                # track most distance neighbors
                m_set = {}
                # track smallest distance of most distant neighbors
                m_val = -float('inf')
                for i in range(d.size):
                    if d[i] > m_val:
                        m_set[d[i]] = i
                        if len(m_set) > self.autoneg_select_ratio:
                            if m_val in m_set:
                                del m_set[m_val]
                        m_val = min(m_set)
                for i in six.itervalues(m_set):
                    neg_autoselect.add(self._descr_cache[i])
            # Remove any positive examples from auto-selected results
            neg_autoselect.difference_update(pos)
            self._log.debug("Auto-selected negative descriptors [%d]: %s",
                            len(neg_autoselect), neg_autoselect)

        num_neg = 0
        for n_iterable in (neg, neg_autoselect):
            for d in n_iterable:
                train_labels.append(-1)
                train_vectors.append(d.vector().tolist())
                num_neg += 1

        if not num_pos:
            raise ValueError("No positive examples provided.")
        elif not num_neg:
            raise ValueError("No negative examples provided.")

        # Training Logistic Regression model
        self._log.debug("online model training")
        param_str = self._gen_lr_parameter_string()
        lr_m = LogisticRegression()
        lr_m = lr_m.set_params(**param_str)
        lr_m.fit(train_vectors, train_labels)
        probs = lr_m.predict_proba(self._descr_matrix)[:,1]
        rank_pool = dict(zip(self._descr_cache, probs))
        return rank_pool

RELEVANCY_INDEX_CLASS = LogisticRegRelevancyIndex
