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

from . import SMQTKClassifier

from SMQTK.utils.FeatureMemory import FeatureMemory


def _svm_model_feature_generator((uid, data, descriptor)):
    """
    :param uid: Data UID
    :type uid: int

    :param data: Data to generate feature over
    :type data: DataFile

    :param descriptor: Feature descriptor that will generate the feature
    :type descriptor: SMQTK.FeatureDescriptors.FeatureDescriptor

    :return: UID and associated feature vector
    :rtype: (int, numpy.ndarray)

    """
    log = logging.getLogger("_svm_model_feature_generator")
    try:
        log.debug("Generating feature for UID[%d] -> %s", uid, data.filepath)
        feat = descriptor.compute_feature(data)
        # Invalid feature matrix if there are inf or NaN values
        # noinspection PyUnresolvedReferences
        if numpy.isnan(feat.sum()):
            return None, None
        return uid, feat
    except Exception, ex:
        log.error("Failed feature generation for data file UID[%d] -> %s\n"
                  "Error: %s",
                  uid, data.filepath, str(ex))
        return None, None


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

    def __init__(self, base_data_dir, base_work_dir, descriptor, ingest):
        super(SVMClassifier_HIK, self).__init__(base_data_dir, base_work_dir,
                                                descriptor, ingest)

        self._ids_filepath = os.path.join(self.data_dir, "id_map.npy")
        self._bg_flags_filepath = os.path.join(self.data_dir, "bg_flags.npy")
        self._feature_data_filepath = os.path.join(self.data_dir, "feature_data.npy")
        self._kernel_data_filepath = os.path.join(self.data_dir, "kernel_data.npy")

        # If we have existing model files, load them into a FeatureMemory
        # instance
        if self.has_model_files():
            self._fm = FeatureMemory.construct_symmetric_from_files()

    # noinspection PyNoneFunctionAssignment,PyUnresolvedReferences,PyTypeChecker
    def generate_model(self, parallel=None):
        """
        Generate this classifiers data-model using the given feature descriptor
        over the configured ingest, saving it to a known location in the
        configured data directory.

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        self.log.debug("Starting model generation")

        parallel = kwds.get('parallel', None)
        if not len(self._ingest):
            raise RuntimeError("Configured ingest has no content!")

        # compute features for ingest content in parallel
        args = []
        for uid, data in self._ingest.iteritems():
            args.append((uid, data, self._descriptor))
        self.log.debug("Processing %d elements", len(args))

        self.log.debug("starting pool...")
        pool = multiprocessing.Pool(processes=parallel)
        map_results = pool.map_async(_svm_model_feature_generator, args).get()
        r_dict = dict(map_results)
        pool.close()
        pool.join()

        # Check for failed generation
        if None in r_dict:
            raise RuntimeError("Failure occurred during model generation. "
                               "See logging.")

        # Initialize data stores
        num_features = len(r_dict)
        sorted_uids = sorted(r_dict.keys())
        feature_length = len(r_dict[sorted_uids[0]])

        idx2uid_map = numpy.empty(num_features, dtype=int)
        idx2bg_map = numpy.empty(num_features, dtype=bool)
        feature_mat = numpy.matrix(numpy.empty((num_features, feature_length),
                                               dtype=float))
        kernel_mat = numpy.matrix(numpy.empty((num_features, num_features),
                                              dtype=float))

        self.log.debug("Building idx2uid map and feature matrix")
        for idx, uid in enumerate(sorted_uids):
            idx2uid_map[idx] = uid
            feature_mat[idx] = r_dict[idx]

        # Flag a percentage of leading data in ingest as background data (auto-
        # negative exemplars). Leading % more deterministic than random for
        # tuning/debugging.
        self.log.debug("Building idx2bg_flags mapping")
        pivot = int(num_features * self.BACKGROUND_RATIO)
        for i in xrange(num_features):
            if i < pivot:
                idx2bg_map[i] = True
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

    def has_model_files(self):
        return (
            os.path.isfile(self._ids_filepath) and
            os.path.isfile(self._bg_flags_filepath) and
            os.path.isfile(self._feature_data_filepath) and
            os.path.isfile(self._kernel_data_filepath)
        )

    def extend_model(self, data):
        """
        Extend the current ingest with new feature data based on the given data
        and the configured feature descriptor, extending the current data model
        for this classifier.

        For not, if there is currently no data model created for this classifier
        / descriptor combination, we will error. In the future, I would imagine
        a new model would be created.

        :param data: Some kind of input data for the feature descriptor. This is
            descriptor dependent.

        """
        # TODO: Assert there there are existing data model files
        raise NotImplementedError()

    def rank_ingest(self, pos_ids, neg_ids):
        """
        Rank the current ingest, returning a mapping of ingest element ID to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1]. Where

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        # TODO: Assert there there are existing data model files
        raise NotImplementedError()


CLASSIFIER_CLASS = SVMClassifier_HIK

