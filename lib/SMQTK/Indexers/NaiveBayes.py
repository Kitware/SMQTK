"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from . import Indexer

import os.path as osp
import numpy
from sklearn.naive_bayes import MultinomialNB

from SMQTK.utils import SimpleTimer


class NaiveBayes_Multinomial (Indexer):

    def __init__(self, data_dir, work_dir):
        super(NaiveBayes_Multinomial, self).__init__(data_dir, work_dir)

        # Array of UIDs in the index the UID refers to in these internal
        # structures
        #: :type: numpy.core.multiarray.ndarray
        self._uid_array = None
        self._uid2idx_map = None

        # Matrix of features
        #: :type: numpy.core.multiarray.ndarray
        self._feature_mat = None

        if self.has_model_files():
            self._load_model_files()

    @property
    def uid_list_filepath(self):
        return osp.join(self.data_dir, "uid_list.npy")

    @property
    def feature_mat_filepath(self):
        return osp.join(self.data_dir, "feature_mat.npy")

    def has_model_files(self):
        return (osp.isfile(self.uid_list_filepath)
                and osp.isfile(self.feature_mat_filepath))

    def _load_model_files(self):
        #: :type: numpy.core.multiarray.ndarray
        self._uid_array = numpy.load(self.uid_list_filepath)
        #: :type: numpy.core.multiarray.ndarray
        self._feature_mat = numpy.load(self.feature_mat_filepath)

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
        )

    def generate_model(self, feature_map, parallel=None, **kwargs):
        """
        Generate this indexers data-model using the given features,
        saving it to files in the configured data directory.

        :raises RuntimeError: Precaution error when there is an existing data
            model for this indexer. Manually delete or move the existing
            model before computing another one.

            Specific implementations may error on other things. See the specific
            implementations for more details.

        :raises ValueError: The given feature map had no content.

        :param feature_map: Mapping of integer IDs to feature data. All feature
            data must be of the same size!
        :type feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        super(NaiveBayes_Multinomial, self).generate_model(feature_map, parallel)

        num_features = len(feature_map)
        ordered_uids = sorted(feature_map.keys())

        sample_feature = feature_map[ordered_uids[0]]
        feature_len = len(sample_feature)

        # Pre-allocating arrays
        self._uid_array = numpy.ndarray(num_features, dtype=int)
        self._feature_mat = numpy.zeros(
            (num_features, feature_len), dtype=sample_feature.dtype
        )

        self.log.info("Populating feature matrix")
        for i, (uid, feat) in enumerate(feature_map.iteritems()):
            self._uid_array[i] = uid
            self._feature_mat[i] = feat

        with SimpleTimer("Saving data files", self.log.info):
            numpy.save(self.uid_list_filepath, self._uid_array)
            numpy.save(self.feature_mat_filepath, self._feature_mat)

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
        super(NaiveBayes_Multinomial, self).extend_model(uid_feature_map, parallel)

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
            self._uid_array.resize((num_features_after,))
            self._feature_mat.resize((num_features_after,
                                      self._feature_mat.shape[1]))

        with SimpleTimer("Adding to matrices", self.log.debug):
            for i in range(num_features_before, num_features_after):
                i_uid = new_uids[i-num_features_before]
                self._uid_array[i] = i_uid
                self._uid2idx_map[i_uid] = i
                self._feature_mat[i] = uid_feature_map[i_uid]

    # def _least_similar_uid(self, uid, N=1):
    #     """
    #     Return an array of N UIDs that are least similar to the feature for the
    #     given UID. If N is greater than the total number of elements in this
    #     indexer's model, we return a list of T ordered elements, where T is
    #     the total number of in the model. I.e. we return an ordered list of all
    #     UIDs by least similarity (the given UID will be the last element in the
    #     list).
    #
    #     :param uid: UID to find the least similar UIDs for.
    #     :type uid: int
    #
    #     :return: List of min(N, T) least similar UIDs.
    #     :rtype: list of int
    #
    #     """
    #     i = self._uid_idx_map[uid]
    #     z = zip(self._uid_array, self._distrance_mat[i])
    #     # Sort by least similarity, pick top N
    #     return [e[0] for e in sorted(z, key=lambda f: f[1], reverse=1)[:N]]
    #
    # def _pick_auto_negatives(self, pos_uids):
    #     """
    #     Pick automatic negative UIDs based on distances from the given positive
    #     UIDs.
    #
    #     :param pos_uids: List of positive UIDs
    #     :type pos_uids: list of int
    #
    #     :return: List of automatically chosen negative UIDs
    #     :rtype: list of int
    #
    #     """
    #     # Pick automatic negatives that are the most distant elements from
    #     # given positive elements.
    #     #: :type: set of int
    #     auto_neg = set()
    #     n = int(self._uid_array.size * self.AUTO_NEG_PERCENT)
    #     for p_UID in pos_uids:
    #         auto_neg.update(self._least_similar_uid(p_UID, n))
    #
    #     # Cancel out any auto-picked negatives that conflict with given positive
    #     # UIDs.
    #     auto_neg.difference_update(pos_uids)
    #
    #     self.log.debug("Post auto-negative selection: %s", auto_neg)
    #     return list(auto_neg)

    def rank_model(self, pos_ids, neg_ids=()):
        super(NaiveBayes_Multinomial, self).rank_model(pos_ids, neg_ids)

        num_pos = len(pos_ids)
        num_neg = len(neg_ids)

        train = numpy.ndarray((num_pos + num_neg, self._feature_mat.shape[1]),
                              dtype=self._feature_mat.dtype)
        train[:num_pos, :] = \
            self._feature_mat[tuple(self._uid2idx_map[uid] for uid in pos_ids), :]
        train[num_pos:num_pos+num_neg, :] = \
            self._feature_mat[tuple(self._uid2idx_map[uid] for uid in neg_ids), :]

        # Positive elements are label 1, negatives are label 0
        labels = numpy.concatenate((numpy.ones(len(pos_ids)),
                                    numpy.zeros(len(neg_ids))))

        # Only really care about probability of positive, so just keeping that
        # column.
        mnb = MultinomialNB()
        probs = mnb.fit(train, labels).predict_proba(self._feature_mat)[:, 1]

        return dict(zip(self._uid_array, probs))

    def reset(self):
        """
        Reset this indexer to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        super(NaiveBayes_Multinomial, self).reset()
        self._load_model_files()


INDEXER_CLASS = [
    NaiveBayes_Multinomial
]