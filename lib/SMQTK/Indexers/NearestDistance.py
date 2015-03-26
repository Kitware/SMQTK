"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import multiprocessing
from operator import neg
import numpy
import os.path as osp

from SMQTK.Indexers import Indexer
from SMQTK.utils import SimpleTimer
from SMQTK.utils.distance_functions import histogram_intersection_distance


class NearestDistance_HIK (Indexer):
    """
    Indexer that ranks elements based on HIK distance metric to other elements
    in the model.
    """

    # Number, in percent to total mdoel elements, of least-similar elements to
    # choose, based on HI distnace, for each positive-labeled element during
    # ranking.
    AUTO_NEG_PERCENT = 0.1

    def __init__(self, data_dir, work_dir):
        super(NearestDistance_HIK, self).__init__(data_dir, work_dir)

        # Array of UIDs in the index the UID refers to in these internal
        # structures
        #: :type: numpy.core.multiarray.ndarray
        self._uid_array = None

        # Matrix of features. This is retained so that we can extend the
        # distance kernel for new features.
        #: :type: numpy.core.multiarray.ndarray
        self._feature_mat = None

        # Distance kernel matrix
        #: :type: numpy.core.multiarray.ndarray
        self._kernel_mat = None

        if self.has_model_files():
            self._load_model_files()

    def _load_model_files(self):
        #: :type: numpy.core.multiarray.ndarray
        self._uid_array = numpy.load(self.uid_list_filepath)
        #: :type: numpy.core.multiarray.ndarray
        self._feature_mat = numpy.load(self.feature_mat_filepath)
        #: :type: numpy.core.multiarray.ndarray
        self._kernel_mat = numpy.load(self.kernel_mat_filepath)

        # Mapping of element UID to array/matrix index position
        #: :type: dict of int
        self._uid_idx_map = {}
        for idx, uid in enumerate(self._uid_array):
            self._uid_idx_map[uid] = idx

    @property
    def uid_list_filepath(self):
        return osp.join(self.data_dir, "uid_list.npy")

    @property
    def feature_mat_filepath(self):
        return osp.join(self.data_dir, "feature_mat.npy")

    @property
    def kernel_mat_filepath(self):
        return osp.join(self.data_dir, 'hik_distance_kernel.npy')

    def has_model_files(self):
        return (osp.isfile(self.uid_list_filepath)
                and osp.isfile(self.feature_mat_filepath)
                and osp.isfile(self.kernel_mat_filepath))

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
            and self._kernel_mat is not None
            and 0 not in self._kernel_mat.shape   # has dimensionality
        )

    def generate_model(self, feature_map, parallel=None):
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
        super(NearestDistance_HIK, self).generate_model(feature_map, parallel)

        num_features = len(feature_map)
        ordered_uids = sorted(feature_map.keys())

        sample_feature = feature_map[ordered_uids[0]]
        feature_len = len(sample_feature)

        # Pre-allocating distance matrix
        uid_array = numpy.ndarray(num_features, dtype=int)
        feature_mat = numpy.zeros(
            (num_features, feature_len), dtype=sample_feature.dtype
        )
        dist_kernel_mat = numpy.zeros(
            (num_features, num_features), dtype=sample_feature.dtype
        )

        self.log.debug("Populating feature matrix")
        for i, (uid, feat) in enumerate(feature_map.iteritems()):
            uid_array[i] = uid
            feature_mat[i] = feat

        self.log.debug("Spawning HI computation tasks")
        pool = multiprocessing.Pool(processes=parallel)
        rmap = {}
        for i in range(num_features):
            for j in range(i, num_features):
                rmap[i, j] = pool.apply_async(histogram_intersection_distance,
                                              args=(feature_mat[i],
                                                    feature_mat[j]))
        # Poll for results in upper triangle of matrix first, as that follows
        # the line of jobs spawned
        with SimpleTimer("Aggregating HI dists into matrix", self.log.debug):
            for i in range(num_features):
                for j in range(i, num_features):
                    dist_kernel_mat[i, j] = rmap[i, j].get()
                    if i != j:
                        dist_kernel_mat[j, i] = dist_kernel_mat[i, j]
        # Filling in top then bottom might result in less cache swapping, but
        # I'm not even sure if this optimization would do anything with python
        # being so high level.
        # with SimpleTimer("Filling in bottom triangle", self.log.debug):
        #     for i in range(1, num_features):
        #         for j in range(0, i):
        #             dist_kernel_mat[i, j] = dist_kernel_mat[j, i]

        self._uid_array = uid_array
        self._feature_mat = feature_mat
        self._kernel_mat = dist_kernel_mat

        with SimpleTimer("Saving data files", self.log.debug):
            numpy.save(self.uid_list_filepath, uid_array)
            numpy.save(self.feature_mat_filepath, feature_mat)
            numpy.save(self.kernel_mat_filepath, dist_kernel_mat)

    def extend_model(self, uid_feature_map, parallel=None):
        """
        Extend, in memory, the current model with the given data elements using
        the configured feature descriptor. Online extensions are not saved to
        data files.

        NOTE: For now, if there is currently no data model created for this
        indexer / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: No current model.

        :raises ValueError: Raised when:

            -   One or more data elements have UIDs that already exist in the
                model.
            -   One or more features are not of the proper shape for this
                indexer's model.

        :param uid_feature_map: Mapping of integer IDs to features to extend this
            indexer's model with.
        :type uid_feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores. Not all implementation support parallel model extension.
        :type parallel: int

        """
        super(NearestDistance_HIK, self).extend_model(uid_feature_map, parallel)

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
        new_uids = uid_feature_map.keys()

        num_features_before = self._feature_mat.shape[0]
        num_features_after = num_features_before + len(uid_feature_map)

        with SimpleTimer("Resizing uid/feature matrices", self.log.debug):
            self._uid_array.resize((num_features_after,))
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
        with SimpleTimer("Submitting HI dist work", self.log.debug):
            for r in range(num_features_before, num_features_after):
                r_uid = new_uids[r-num_features_before]
                self._uid_array[r] = r_uid
                self._uid_idx_map[r_uid] = r
                self._feature_mat[r] = uid_feature_map[r_uid]
                for c in range(r+1):
                    hid_map[r, c] = pool.apply_async(
                        histogram_intersection_distance,
                        args=(self._feature_mat[r], self._feature_mat[c])
                    )
        pool.close()

        # expanding kernel matrix
        # noinspection PyNoneFunctionAssignment
        with SimpleTimer("'Resizing' kernel matrix", self.log.debug):
            new_km = numpy.ndarray((num_features_after, num_features_after),
                                   dtype=self._kernel_mat.dtype)
            new_km[:num_features_before,
                   :num_features_before] = self._kernel_mat
            self._kernel_mat = new_km

        with SimpleTimer("Collecting dist results into matrix", self.log.debug):
            for (r, c), dist in hid_map.iteritems():
                d = dist.get()
                self._kernel_mat[r, c] = d
                self._kernel_mat[c, r] = d

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
        i = self._uid_idx_map[uid]
        z = zip(self._uid_array, self._kernel_mat[i])
        # Sort by least similarity, pick top N
        return [e[0] for e in sorted(z, key=lambda f: f[1])[:N]]

    def rank_model(self, pos_ids, neg_ids=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1], where 1.0 is the highest rank and 0.0 is the lowest rank.

        :raises RuntimeError: No current model.

        :param pos_ids: List of positive data IDs
        :type pos_ids: collections.Iterable of int

        :param neg_ids: List of negative data IDs
        :type neg_ids: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        super(NearestDistance_HIK, self).rank_model(pos_ids, neg_ids)

        # # Pick automatic negatives that are the most distant elements from given
        # # positive elements.
        # #: :type: set of int
        # new_neg = set()
        # n = int(self._uid_array.size * self.AUTO_NEG_PERCENT)
        # for p_UID in pos_ids:
        #     new_neg.update(self._least_similar_uid(p_UID, n))
        # #: :type: set of int
        # neg_ids = set(neg_ids).union(new_neg.difference(pos_ids))

        pidx = [self._uid_idx_map[pid] for pid in pos_ids]
        nidx = [self._uid_idx_map[nid] for nid in neg_ids]
        npow = numpy.power
        d = 1.0 - self._kernel_mat  # kernel is similarity matrix at the moment

        # get the squared distances for positive and negative rows
        pos_sqrd_dists = npow(d[pidx, :], 2.0)
        neg_sqrd_dists = npow(d[nidx, :], 2.0)
        # sum both along rows to make (1xN) vector
        pos_dist = pos_sqrd_dists.sum(axis=0)
        neg_dist = neg_sqrd_dists.sum(axis=0)
        # subtract negative dist vector from positive to weight highly distant
        # elements away.
        dists = pos_dist - neg_dist

        # Shift minimum to 0-point, scale result into [0,1] scale as per comment
        dists -= dists.min()
        rank = 1.0 - (dists / dists.max())

        # dists vector still in column order of kernel, so pair with UID vector
        # for result ranking map.
        return dict(zip(self._uid_array, rank))

    def reset(self):
        """
        Reset this indexer to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        super(NearestDistance_HIK, self).reset()

        # Reloading data elements from file
        self._load_model_files()


INDEXER_CLASS = NearestDistance_HIK
