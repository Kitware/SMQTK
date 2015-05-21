"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import cPickle
import logging
import multiprocessing.pool
import numpy
import os.path as osp
import svm
import svmutil

import smqtk_config

from smqtk.indexing import Indexer
from smqtk.utils import safe_create_dir, SimpleTimer
from smqtk.utils.distance_functions import histogram_intersection_distance


def _svm_model_hik_helper(i, j, i_feat, j_feat):
    """
    HIK async compute helper
    """
    log = logging.getLogger("_svm_model_hik_helper")
    log.debug("Computing HIK for [%d, %d]", i, j)
    # noinspection PyUnresolvedReferences
    ij_hik = histogram_intersection_distance(i_feat, j_feat)
    return ij_hik


class SVMIndexerHIK (Indexer):
    """
    Indexer using SVM classification model with Platt scaling

    Inherited from progenitor ALADDIN project

    """

    # Pick lowest % intersecting elements as auto-bg
    AUTO_NEG_PERCENT = 0.10

    def __init__(self, data_dir):
        self.data_dir = osp.join(smqtk_config.DATA_DIR, data_dir)

        # Array of UIDs in the index the UID refers to in these internal
        # structures
        #: :type: list[collections.Hashable]
        self._uid_array = None
        #: :type: dict[object, int]
        self._uid2idx_map = None

        # Matrix of features
        #: :type: numpy.core.multiarray.ndarray
        self._feature_mat = None

        # Distance kernel matrix
        #: :type: numpy.core.multiarray.ndarray
        self._distance_mat = None

        # Templated to take the W1 integer weight argument, which should be
        # floor(num_negatives / num_positives), which a min value of 1
        # - The ``-t 5`` parameter value is unique to the custom libSVM
        #   implementation we build.
        self.svm_train_params = '-q -t 5 -b 1 -c 2 -w1 %f -g 0.0078125'

        if self.has_model_files():
            self._load_model_files()

    @property
    def uid_list_filepath(self):
        return osp.join(self.data_dir, "uid_list.pickle")

    @property
    def feature_mat_filepath(self):
        return osp.join(self.data_dir, "feature_mat.npy")

    @property
    def distance_mat_filepath(self):
        return osp.join(self.data_dir, 'hik_distance_kernel.npy')

    def has_model_files(self):
        return (osp.isfile(self.uid_list_filepath)
                and osp.isfile(self.feature_mat_filepath)
                and osp.isfile(self.distance_mat_filepath))

    def _load_model_files(self):
        with open(self.uid_list_filepath, 'rb') as ifile:
            #: :type: list[collections.Hashable]
            self._uid_array = cPickle.load(ifile)
        #: :type: numpy.core.multiarray.ndarray
        self._feature_mat = numpy.load(self.feature_mat_filepath)
        #: :type: numpy.core.multiarray.ndarray
        self._distance_mat = numpy.load(self.distance_mat_filepath)

        # Mapping of element UID to array/matrix index position
        #: :type: dict of int
        self._uid2idx_map = {}
        for idx, uid in enumerate(self._uid_array):
            self._uid2idx_map[uid] = idx

    def has_model(self):
        """
        :return: True if this indexer has a valid initialized model for
            extension and ranking (or doesn't need one to perform those tasks).
        :rtype: bool
        """
        return (
            self._uid_array is not None
            and self._feature_mat is not None
            and 0 not in self._feature_mat.shape  # has dimensionality
            and self._distance_mat is not None
            and 0 not in self._distance_mat.shape   # has dimensionality
        )

    def generate_model(self, descriptor_map, parallel=None, **kwargs):
        """
        Generate this indexers data-model using the given features,
        saving it to files in the configured data directory.

        :raises RuntimeError: Precaution error when there is an existing data
            model for this indexer. Manually delete or move the existing
            model before computing another one.

            Specific implementations may error on other things. See the specific
            implementations for more details.

        :raises ValueError: The given feature map had no content.

        :param descriptor_map: Mapping of integer IDs to feature data. All feature
            data must be of the same size!
        :type descriptor_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        if self.has_model():
            raise RuntimeError("WARNING: This implementation already has a "
                               "model generated. These can take a long time to "
                               "generate, thus we require external manual "
                               "removal of modal files before we will generate "
                               "a new model.")

        num_features = len(descriptor_map)
        ordered_uids = sorted(descriptor_map.keys())

        sample_feature = descriptor_map[ordered_uids[0]]
        feature_len = len(sample_feature)

        # Pre-allocating arrays
        #: :type: list[collections.Hashable]
        self._uid_array = []
        self._feature_mat = numpy.zeros(
            (num_features, feature_len), dtype=sample_feature.dtype
        )
        self._distance_mat = numpy.zeros(
            (num_features, num_features), dtype=sample_feature.dtype
        )

        with SimpleTimer("Populating feature matrix", self.log.info):
            for i, (uid, feat) in enumerate(descriptor_map.iteritems()):
                self._uid_array.append(uid)
                self._feature_mat[i] = feat

        with SimpleTimer("Computing HI matrix kernel", self.log.info):
            # Using [process] Pool here with large sets eats far too much RAM.
            # Using a ThreadPool here is actually much slower. Not sure why?
            for i in range(num_features):
                for j in range(i, num_features):
                    self._distance_mat[i, j] = self._distance_mat[j, i] = \
                        histogram_intersection_distance(self._feature_mat[i],
                                                        self._feature_mat[j])

        with SimpleTimer("Saving data files", self.log.info):
            safe_create_dir(self.data_dir)
            with open(self.uid_list_filepath, 'wb') as ofile:
                cPickle.dump(self._uid_array, ofile)
            numpy.save(self.feature_mat_filepath, self._feature_mat)
            numpy.save(self.distance_mat_filepath, self._distance_mat)
            # TODO: destruct and reload matrices in memmap mode
            #       - see numpy.load() doc-string

    def extend_model(self, uid_feature_map, parallel=None):
        """
        Extend, in memory, the current model with the given feature elements.
        Online extensions are not saved to data files.

        NOTE: For now, if there is currently no data model created for this
        indexer / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: No current model.

        :param uid_feature_map: Mapping of integer IDs to features to extend this
            indexer's model with.
        :type uid_feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores. Not all implementation support parallel model extension.
        :type parallel: int

        """
        if not self.has_model():
            raise RuntimeError("No model available for this indexer.")

        # Shortcut when we're not given anything to actually process
        if not uid_feature_map:
            self.log.debug("No new features to extend")
            return

        # Check UID intersection
        with SimpleTimer("Checking UID uniqueness", self.log.debug):
            cur_uids = set(self._uid_array)
            intersection = cur_uids.intersection(uid_feature_map.keys())
            if intersection:
                raise ValueError("The following IDs are already present in the "
                                 "indexer's model: %s" % tuple(intersection))

        # Check feature consistency
        # - Assuming that there is are least one feature in our current model...
        with SimpleTimer("Checking input feature shape", self.log.debug):
            example_feat = self._feature_mat[0]
            for feat in uid_feature_map.values():
                if feat.shape[0] != example_feat.shape[0]:
                    raise ValueError("One or more features provided are not of "
                                     "the correct shape! Found %s when we "
                                     "require %s"
                                     % (feat.shape, example_feat.shape[1]))
            del example_feat  # Deleting so we can resize later in the function

        # Extend data structures
        # - UID and Feature matrix can be simply resized in-place as we are
        #   strictly adding to the end of the structure in memory.
        # - distance matrix, since we're adding new columns in addition to rows,
        #   need to create a new matrix of the desired shape, copying in
        #   existing into new matrix.
        self.log.debug("Sorting feature UIDs")
        new_uids = sorted(uid_feature_map.keys())

        self.log.debug("Calculating before and after sizes.")
        num_features_before = self._feature_mat.shape[0]
        num_features_after = num_features_before + len(uid_feature_map)

        with SimpleTimer("Resizing uid/feature matrices", self.log.debug):
            self._feature_mat.resize((num_features_after,
                                      self._feature_mat.shape[1]))

        # Calculate distances for new features to all others
        # - for-each new feature row, calc distance to all features in rows
        #   before it + itself
        # - r is the index of the current new feature
        #       (num_features_before <= r < num_features_after)
        #   c is the index of the feature we are computing the distance to
        #       (0 <= c <= r)
        # - Expanding and copying kernel matrix while computing distance to not
        #   waste time waiting for computations to finish
        pool = multiprocessing.Pool(processes=parallel)
        hid_map = {}
        with SimpleTimer("Adding to matrices, submitting HI work",
                         self.log.debug):
            for r in range(num_features_before, num_features_after):
                r_uid = new_uids[r-num_features_before]
                self._uid_array.append(r_uid)
                assert len(self._uid_array) == r+1
                self._uid2idx_map[r_uid] = r
                self._feature_mat[r] = uid_feature_map[r_uid]
                for c in range(r+1):
                    hid_map[r, c] = pool.apply_async(
                        histogram_intersection_distance,
                        args=(self._feature_mat[r], self._feature_mat[c])
                    )
        pool.close()

        # Expanding kernel matrix in local memory while async processing is
        # going on.
        # noinspection PyNoneFunctionAssignment
        with SimpleTimer("'Resizing' kernel matrix", self.log.debug):
            new_dm = numpy.ndarray((num_features_after, num_features_after),
                                   dtype=self._distance_mat.dtype)
            new_dm[:num_features_before,
                   :num_features_before] = self._distance_mat
            self._distance_mat = new_dm

        with SimpleTimer("Collecting dist results into matrix", self.log.debug):
            for (r, c), dist in hid_map.iteritems():
                d = dist.get()
                self._distance_mat[r, c] = self._distance_mat[c, r] = d
        pool.join()

    def _least_similar_uid(self, uid, N=1):
        """
        Return an array of N UIDs that are least similar to the feature for the
        given UID. If N is greater than the total number of elements in this
        indexer's model, we return a list of T ordered elements, where T is
        the total number of in the model. I.e. we return an ordered list of all
        UIDs by least similarity (the given UID will be the last element in the
        list).

        :param uid: UID to find the least similar UIDs for.
        :type uid: int

        :return: List of min(N, T) least similar UIDs.
        :rtype: list of int

        """
        i = self._uid2idx_map[uid]
        z = zip(self._uid_array, self._distance_mat[i])
        # Sort by least similarity, pick top N
        return [e[0] for e in sorted(z, key=lambda f: f[1], reverse=1)[:N]]

    def _pick_auto_negatives(self, pos_uids):
        """
        Pick automatic negative UIDs based on distances from the given positive
        UIDs.

        :param pos_uids: List of positive UIDs
        :type pos_uids: list of int

        :return: List of automatically chosen negative UIDs
        :rtype: set of int

        """
        # Pick automatic negatives that are the most distant elements from
        # given positive elements.
        #: :type: set of int
        auto_neg = set()
        n = max(1, int(len(self._uid_array) * self.AUTO_NEG_PERCENT))
        for p_UID in pos_uids:
            auto_neg.update(self._least_similar_uid(p_UID, n))

        # Cancel out any auto-picked negatives that conflict with given positive
        # UIDs.
        auto_neg.difference_update(pos_uids)

        self.log.debug("Post auto-negative selection: %s", auto_neg)
        return auto_neg

    def rank_model(self, pos_ids, neg_ids=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1], where 1.0 is the highest rank and 0.0 is the lowest rank.

        :raises RuntimeError: No current model.

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        :param pos_ids: List of positive data IDs. Required.
        :type pos_ids: list of int

        :param neg_ids: List of negative data IDs. Optional.
        :type neg_ids: list of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        if not self.has_model():
            raise RuntimeError("No model available for this indexer.")

        # Automatically support the negative IDs with the most distance UIDs
        # from the provided positive UIDs.
        # if len(neg_ids) == 0:
        #     neg_ids = self._pick_auto_negatives(pos_ids)
        neg_ids = set(neg_ids).union(self._pick_auto_negatives(pos_ids))

        #
        # SVM model training
        #
        uid_list = sorted(set.union(set(pos_ids), neg_ids))
        feature_len = self._feature_mat.shape[1]
        # positive label: 1, negative label: 0
        bool2label = {1: 1, 0: 0}
        labels = [bool2label[uid in pos_ids] for uid in uid_list]
        train_features = \
            self._feature_mat[list(self._uid2idx_map[uid] for uid in uid_list), :]

        self.log.debug("Creating SVM problem")
        svm_problem = svm.svm_problem(labels, train_features.tolist())
        self.log.debug("Creating SVM model")
        w1_weight = max(1.0, len(neg_ids)/float(len(pos_ids)))
        svm_model = svmutil.svm_train(svm_problem,
                                      self.svm_train_params % w1_weight)
        if svm_model.l == 0:
            raise RuntimeError("SVM Model learning failed")

        # Finding associated clip IDs of trained support vectors
        self.log.debug("Finding clip IDs for support vectors")
        hash2feature_idx = dict([(hash(tuple(f)), r)
                                 for r, f in enumerate(self._feature_mat)])
        svm_sv_idxs = []
        tmp_list = [0] * feature_len
        for r in range(svm_model.nSV[0] + svm_model.nSV[1]):
            for c in range(feature_len):
                tmp_list[c] = svm_model.SV[r][c].value
            svm_sv_idxs.append(hash2feature_idx[hash(tuple(tmp_list))])

        #
        # Platt Scaling for probability ranking
        #

        # Features associated to support vectors in trained model
        self.log.debug("Forming data for Platt Scaling")
        # We need the distances between support vectors to all features
        test_kernel = self._distance_mat[svm_sv_idxs, :]

        weights = numpy.array(svm_model.get_sv_coef()).flatten()
        margins = (numpy.mat(weights) * test_kernel).A[0]

        self.log.debug("Performing Platt scaling")
        rho = svm_model.rho[0]
        probA = svm_model.probA[0]
        probB = svm_model.probB[0]
        #: :type: numpy.core.multiarray.ndarray
        probs = 1.0 / (1.0 + numpy.exp((margins - rho) * probA + probB))

        # Test if the probability of an adjudicated positive is below a
        # threshold. If it is, invert probabilities.
        # * Find lowest ranking positive example
        # * Test if the probability valuation falls in the lower 50% of all
        #   probabilities.
        pos_probs = numpy.array(
            [probs[self._uid2idx_map[uid]] for uid in pos_ids]
        )
        pos_mean_prob = pos_probs.sum() / pos_probs.size
        total_mean_prob = probs.sum() / probs.size
        if pos_mean_prob < total_mean_prob:
            probs = 1.0 - probs

        probability_map = dict(zip(self._uid_array, probs))

        return probability_map

    def reset(self):
        """
        Reset this indexer to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        if not self.has_model():
            raise RuntimeError("No model available for this indexer to reset "
                               "to.")
        self._load_model_files()


INDEXER_CLASS = SVMIndexerHIK
