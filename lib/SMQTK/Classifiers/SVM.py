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

from SMQTK.Classifiers import SMQTKClassifier
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


class SVMClassifier_HIK (SMQTKClassifier):
    """
    SVM classifier implementation
    """

    BACKGROUND_RATIO = 0.40  # first 40%

    def __init__(self, data_dir, work_dir):
        super(SVMClassifier_HIK, self).__init__(data_dir, work_dir)

        self._ids_filepath = os.path.join(self.data_dir, "id_map.npy")
        self._bg_flags_filepath = os.path.join(self.data_dir, "bg_flags.npy")
        self._feature_data_filepath = os.path.join(self.data_dir, "feature_data.npy")
        self._kernel_data_filepath = os.path.join(self.data_dir, "kernel_data.npy")

        self.svm_train_params = '-q -t 4 -b 1 -w1 50 -c 100'

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
        Generate this classifiers data-model using the given features,
        saving it to files in the configured data directory.

        :raises RuntimeError: Precaution error when there is an existing data
            model for this classifier. Manually delete or move the existing
            model before computing another one.

        :param feature_map: Mapping of integer IDs to feature data. All feature
            data must be of the same size!
        :type feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        if self._feat_mem is not None:
            raise RuntimeError("\n"
                               "!!! Warning !!! Warning !!! Warning !!!\n"
                               "A model already exists for this"
                               "ingest / descriptor combination! Make sure "
                               "that you really want to do this by moving / "
                               "deleting the existing model. Model location: "
                               "%s\n"
                               "!!! Warning !!! Warning !!! Warning !!!"
                               % self.data_dir)

        self.log.debug("Starting model generation")

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

        self.log.debug("Building idx2uid map and feature matrix")
        for idx, uid in enumerate(sorted_uids):
            idx2uid_map[idx] = uid
            feature_mat[idx] = feature_map[idx]

        # Flag a percentage of leading data in ingest as background data (auto-
        # negative exemplars). Leading % more deterministic than random for
        # tuning/debugging.
        self.log.debug("Building idx2bg_flags mapping")
        pivot = int(num_features * self.BACKGROUND_RATIO)
        for i in xrange(num_features):
            if i < pivot:
                idx2bg_map[i] = True
                bg_id_set.add(idx2uid_map[i])
            else:
                idx2bg_map[i] = False

        # Constructing histogram intersection kernel
        self.log.debug("Computing Histogram Intersection kernel matrix")
        pool = multiprocessing.Pool(processes=parallel)
        self.log.debug("Entering jobs...")
        results = {}
        for i, i_feat in enumerate(feature_mat):
            results[i] = {}
            for j, j_feat in enumerate(feature_mat):
                results[i][j] = pool.apply_async(_svm_model_hik_helper,
                                                 (i, j, i_feat, j_feat))
        self.log.debug("Collecting results...")
        for i in results.keys():
            for j in results[i].keys():
                kernel_mat[i, j] = results[i][j].get()
        pool.close()
        pool.join()

        self.log.debug("Saving data files")
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
        :return: If this classifier instance currently has a model loaded.
        :rtype: bool
        """
        return self._feat_mem is not None

    def extend_model(self, id_feature_map, parallel=None):
        """
        Extend, in memory, the current data model with given data elements using
        the configured feature descriptor.

        NOTE: For now, if there is currently no data model created for this
        classifier / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: When there is no existing data model present to
            extend.
        :raises ValueError: Raised when:
            -   One or more data elements have UIDs that already exist in the
                model.
            -   One or more features are not of the proper shape for this
                classifier's model.

        :param id_feature_map: Mapping of integer IDs to features to extend this
            classifier's model with.
        :type id_feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        if self._feat_mem is None:
            raise RuntimeError("No model for this classifier yet! Expected to "
                               "find files at: %s" % self.data_dir)

        with self._feat_mem_lock.write_lock():
            cur_ids = set(self._feat_mem.get_ids())

            # Check UID intersection
            intersection = cur_ids.intersection(id_feature_map.keys())
            if intersection:
                raise ValueError("The following IDs are already present in the "
                                 "classifier's model: %s" % tuple(intersection))

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
            raise RuntimeError("No model for this classifier yet! Expected to "
                               "find files at: %s" % self.data_dir)

        self.log.debug("Extracting symmetric submatrix from DK")
        idx2id_map, idx2isbg_map, m = \
            self._feat_mem.get_distance_kernel().symmetric_submatrix(*pos_ids)
        self.log.debug("-- num bg: %d", idx2isbg_map.count(True))
        self.log.debug("-- m shape: %s", m.shape)

        # for model training function, inverse of idx_is_bg: True
        # indicates a positively adjudicated index
        labels_train = numpy.array(tuple(not b for b in idx2isbg_map))

        #
        # SVM Training
        #

        # Returned dictionary contains the keys "model" and "clipid_SVs"
        # referring to the trained model and a list of support vectors,
        # respectively.
        ret_dict = iqr_model_train(m, labels_train, idx2id_map,
                                   self.svm_train_params)
        svm_model = ret_dict['model']
        svm_svIDs = ret_dict['clipids_SVs']

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

        # Testing/Ranking call
        #   Passing the array version of the kernel sub-matrix. The
        #   returned output['probs'] type matches the type passed in
        #   here, and using an array makes syntax cleaner.
        self.log.debug("Ranking model IDs")
        output = iqr_model_test(svm_model, kernel_test.A, idx2id_col)
        probability_map = dict(zip(output['clipids'], output['probs']))

        # Force adjudicated negatives to be probability 0.0 since we don't
        # want them possibly polluting the further adjudication views.
        for uid in neg_ids:
            probability_map[uid] = 0.0

        return probability_map

    def reset(self):
        """
        Reset this classifier to its original state, i.e. removing any model
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


CLASSIFIER_CLASS = SVMClassifier_HIK
