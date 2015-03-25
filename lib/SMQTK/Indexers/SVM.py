"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging
import multiprocessing
import numpy
import os
import shutil

from EventContentDescriptor.iqr_modules import iqr_model_train, iqr_model_test

from SMQTK.Indexers import Indexer
from SMQTK.utils.FeatureMemory import FeatureMemory
from SMQTK.utils.ReadWriteLock import ReadWriteLock


def _svm_model_hik_helper(i, j, i_feat, j_feat):
    """
    HIK async compute helper
    """
    log = logging.getLogger("_svm_model_hik_helper")
    log.debug("Computing HIK for [%d, %d]", i, j)
    # noinspection PyUnresolvedReferences
    ij_hik = (i_feat + j_feat - numpy.abs(i_feat - j_feat)).sum() * 0.5
    return ij_hik


class SVMIndexer_HIK (Indexer):
    """
    SVM indexer implementation

    Inherited from progenitor ALADDIN project

    """
    # TODO: Add optional global model caching so that successive indexer
    #       construction doesn't take additional time.

    BACKGROUND_RATIO = 0.40  # first 40%

    # Pick lowest 20% intersecting elements as auto-bg
    AUTO_BG_PERCENT = 0.20

    def __init__(self, data_dir, work_dir):
        super(SVMIndexer_HIK, self).__init__(data_dir, work_dir)

        self._ids_filepath = os.path.join(self.data_dir, "id_map.npy")
        self._bg_flags_filepath = os.path.join(self.data_dir, "bg_flags.npy")
        self._feature_data_filepath = os.path.join(self.data_dir, "feature_data.npy")
        self._kernel_data_filepath = os.path.join(self.data_dir, "kernel_data.npy")

        self.svm_train_params = '-q -t 4 -b 1 -w1 8 -c 20'
        # self.svm_train_params = '-q -t 4 -b 1 -w1 8 -c 1'
        # self.svm_train_params = '-q -t 0 -b 1 -w1 8 -c 1'

        # If we have existing model files, load them into a FeatureMemory
        # instance
        self._feat_mem_lock = ReadWriteLock()
        self._feat_mem = None
        # Not using the reset method here as we need to allow for object
        # construction so as to be able to call the ``generate_model`` method.
        if self._has_model_files():
            self._feat_mem = FeatureMemory.construct_from_files(
                self._ids_filepath, self._bg_flags_filepath,
                self._feature_data_filepath, self._kernel_data_filepath,
                rw_lock=self._feat_mem_lock
            )

    # noinspection PyNoneFunctionAssignment,PyUnresolvedReferences,PyTypeChecker
    def generate_model(self, feature_map, parallel=None):
        """
        Generate this indexers data-model using the given features,
        saving it to files in the configured data directory.

        :raises RuntimeError: Precaution error when there is an existing data
            model for this indexer. Manually delete or move the existing
            model before computing another one.

        :raises ValueError: The given feature map had no content.

        :param feature_map: Mapping of integer IDs to feature data. All feature
            data must be of the same size!
        :type feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        super(SVMIndexer_HIK, self).generate_model(feature_map, parallel)

        self.log.info("Starting model generation")

        # Initialize data stores
        num_features = len(feature_map)
        sorted_uids = sorted(feature_map.keys())
        feature_length = len(feature_map[sorted_uids[0]])

        idx2uid_map = numpy.empty(num_features, dtype=int)
        idx2bg_map = numpy.empty(num_features, dtype=bool)
        bg_id_set = set()
        feature_mat = numpy.matrix(numpy.empty((num_features, feature_length),
                                               dtype=float))
        kernel_mat = numpy.matrix(numpy.empty((num_features, num_features),
                                              dtype=float))

        self.log.info("Building idx2uid map and feature matrix")
        for idx, uid in enumerate(sorted_uids):
            idx2uid_map[idx] = uid
            feature_mat[idx] = feature_map[uid]

        # Flag a percentage of leading data in ingest as background data (auto-
        # negative exemplars). Leading % more deterministic than random for
        # tuning/debugging.
        self.log.info("Building idx2bg_flags mapping")
        pivot = int(num_features * self.BACKGROUND_RATIO)
        for i in xrange(num_features):
            if i < pivot:
                idx2bg_map[i] = True
                bg_id_set.add(idx2uid_map[i])
            else:
                idx2bg_map[i] = False

        # Constructing histogram intersection kernel
        self.log.info("Computing Histogram Intersection kernel matrix")
        pool = multiprocessing.Pool(processes=parallel)
        self.log.info("\tEntering jobs...")
        results = {}
        for i, i_feat in enumerate(feature_mat):
            results[i] = {}
            for j, j_feat in enumerate(feature_mat):
                results[i][j] = pool.apply_async(_svm_model_hik_helper,
                                                 (i, j, i_feat, j_feat))
        self.log.info("\tCollecting results...")
        for i in results.keys():
            # self.log.info("\t\tRow: %d/%d", i, len(results)-1)
            for j in results[i].keys():
                kernel_mat[i, j] = results[i][j].get()
        pool.close()
        pool.join()

        self.log.info("Saving data files")
        numpy.save(self._ids_filepath, idx2uid_map)
        numpy.save(self._bg_flags_filepath, idx2bg_map)
        numpy.save(self._feature_data_filepath, feature_mat)
        numpy.save(self._kernel_data_filepath, kernel_mat)

    def _has_model_files(self):
        return (
            os.path.isfile(self._ids_filepath) and
            os.path.isfile(self._bg_flags_filepath) and
            os.path.isfile(self._feature_data_filepath) and
            os.path.isfile(self._kernel_data_filepath)
        )

    def has_model(self):
        """
        :return: True if this indexer has a valid initialized model for
            extension and ranking (or doesn't need one to perform those tasks).
        :rtype: bool
        """
        return self._feat_mem is not None

    def extend_model(self, id_feature_map, parallel=None):
        """
        Extend, in memory, the current data model with given data elements using
        the configured feature descriptor.

        NOTE: For now, if there is currently no data model created for this
        indexer / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: When there is no existing data model present to
            extend.
        :raises ValueError: Raised when:
            -   One or more data elements have UIDs that already exist in the
                model.
            -   One or more features are not of the proper shape for this
                indexer's model.

        :param id_feature_map: Mapping of integer UIDs to features to extend
            this indexer's model with.
        :type id_feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        if self._feat_mem is None:
            raise RuntimeError("No model for this indexer yet! Expected to "
                               "find files at: %s" % self.data_dir)

        with self._feat_mem_lock.write_lock():
            cur_ids = set(self._feat_mem.get_ids())

            # Check UID intersection
            intersection = cur_ids.intersection(id_feature_map.keys())
            if intersection:
                raise ValueError("The following IDs are already present in the "
                                 "indexer's model: %s" % tuple(intersection))

            # Check feature consistency
            example_feat = self._feat_mem.get_feature_matrix()[0]
            for feat in id_feature_map.values():
                if feat.shape[0] != example_feat.shape[1]:
                    raise ValueError("One or more features provided are not of "
                                     "the correct shape! Found %s when we "
                                     "require %s"
                                     % (feat.shape, example_feat.shape[1]))

            # Add computed features to FeatureMemory
            for uid, feat in id_feature_map.iteritems():
                self.log.debug("Updating FeatMem with data: %s",
                               id_feature_map[uid])
                self._feat_mem.update(uid, feat)

    def rank_model(self, pos_ids, neg_ids=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1]. Where

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        :param pos_ids: List of positive data IDs. Required.
        :type pos_ids: list of int

        :param neg_ids: List of negative data IDs. Optional.
        :type neg_ids: list of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        if self._feat_mem is None:
            raise RuntimeError("No model for this indexer yet! Expected to "
                               "find files at: %s" % self.data_dir)

        # === EXPERIMENT ===
        # Swapping out FeatureMemory background clips with auto-selected UIDs
        # that are those that intersect least with the given positive UIDs.
        # NOTE: There must be at least 1/AUTO_BG_SELECTOR_RATIO elements in the
        #       model for this to make sense.
        self.log.debug("Auto-selecting additional negative UID")
        self.log.debug("Selecting %2.2f%% least similar", self.AUTO_BG_PERCENT)
        original_bgUIDs = self._feat_mem._bg_clip_ids
        autoselected_neg_UIDs = set()
        for pos_UID in pos_ids:
            # Using the row of the DK corresponding to the UID, sort
            # (index, intersection) pairs in ascending intersection and store
            # the UIDs of the top X% of least intersecting points.
            _, colUIDs, m = \
                self._feat_mem.get_distance_kernel().extract_rows(pos_UID)
            num_least_intersecting = int(len(colUIDs) * self.AUTO_BG_PERCENT)
            # self.log.debug("HIK distance matrix: %s", m)
            # List of (UID, intersection) tuples in order of ascending
            # intersection value. Got one row, so morphing matrix into 1D array.
            ordered_HIK = sorted(zip(colUIDs, m.A[0]), key=lambda e: e[1])
            # self.log.debug("Ordered pairs: %s",
            #                ordered_HIK[:num_least_intersecting])
            # list of least intersecting UIDs
            least_intersecting_uids = \
                [elem[0] for elem
                 in ordered_HIK[:num_least_intersecting]]
            # self.log.debug("%.2f%% Least similar to PosID[%d]: %s",
            #                self.AUTO_BG_PERCENT * 100, pos_UID,
            #                least_intersecting_uids)
            autoselected_neg_UIDs.update(least_intersecting_uids)
        # User supplied positives override auto or supplied negatives
        neg_ids = autoselected_neg_UIDs.union(neg_ids).difference(pos_ids)
        self._feat_mem._bg_clip_ids = neg_ids
        self.log.debug("Updated Pos/Neg:\n"
                       "Pos: %s\n"
                       "Neg: %s",
                       pos_ids, neg_ids)

        #
        # SVM Training
        #
        self.log.debug("Extracting symmetric submatrix from DK")
        idx2id_map, idx2isbg_map, m = \
            self._feat_mem.get_distance_kernel().symmetric_submatrix(*pos_ids)
        m = 1.0 - m  # inverse to get distance instead of similarity
        self.log.debug("-- num bg: %d", idx2isbg_map.count(True))
        self.log.debug("-- m shape: %s", m.shape)

        # for model training function, inverse of idx_is_bg: True
        # indicates a positively adjudicated index
        labels_train = numpy.array(tuple(not b for b in idx2isbg_map))

        # Returned dictionary contains the keys "model" and "clipid_SVs"
        # referring to the trained model and a list of support vectors,
        # respectively.
        ret_dict = iqr_model_train(m, labels_train, idx2id_map,
                                   self.svm_train_params)
        svm_model = ret_dict['model']
        svm_svIDs = ret_dict['clipids_SVs']
        svm_svID_labels = [svID in pos_ids for svID in svm_svIDs]

        #
        # SVM Model application ("testing")
        #
        self.log.info("Starting model application...")

        # As we're extracting rows, the all IDs in the model are preserved along
        # the x-axis (column IDs). The list of IDs along the x-axis is
        # thus effectively the ordered list of all IDs.
        idx2id_row, idx2id_col, kernel_test = \
            self._feat_mem.get_distance_kernel()\
                          .extract_rows(*svm_svIDs)
        kernel_test = 1.0 - kernel_test  # inverse to get distance instead of similarity

        # Testing/Ranking call
        #   Passing the array version of the kernel sub-matrix. The
        #   returned output['probs'] type matches the type passed in
        #   here, and using an array makes syntax cleaner.
        self.log.debug("Ranking model IDs")
        output = iqr_model_test(svm_model, kernel_test.A, idx2id_col,
                                svm_svID_labels)
        probability_map = dict(zip(output['clipids'], output['probs']))

        # Restoring feature memories original background clip IDs
        self._feat_mem._bg_clip_ids = original_bgUIDs

        return probability_map

    def reset(self):
        """
        Reset this indexer to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: There are no current model files to reset to.

        """
        if not self._has_model_files():
            raise RuntimeError("No model files to reset state to! Please "
                               "generate first.")
        self._feat_mem = FeatureMemory.construct_from_files(
            self._ids_filepath, self._bg_flags_filepath,
            self._feature_data_filepath, self._kernel_data_filepath,
            rw_lock=self._feat_mem_lock
        )


INDEXER_CLASS = SVMIndexer_HIK
