# coding=utf-8

import logging
import numpy
import multiprocessing
import multiprocessing.pool
import os
import os.path as osp
import shutil
import uuid

from SMQTK.utils import safe_create_dir


class IqrResultsDict (dict):
    """
    Dictionary subclass for standardizing data types stored.
    """

    def __setitem__(self, i, v):
        super(IqrResultsDict, self).__setitem__(int(i), float(v))

    def update(self, other=None, **kwds):
        """
        D.update([E, ]**F) -> None. Update D from dict/iterable E and F.
        If E present and has a .keys() method, does: for k in E: D[k] = E[k]
        If E present and lacks .keys() method, does: for (k, v) in E: D[k] = v
        In either case, this is followed by: for k in F: D[k] = F[k]

        Reimplemented so as to use override __setitem__ method.
        """
        if hasattr(other, 'keys'):
            for k in other:
                self[k] = other[k]
        else:
            for k, v in other:
                self[k] = v
        for k in kwds:
            self[k] = kwds[k]


class IqrSession (object):

    @property
    def _log(self):
        return logging.getLogger(
            '.'.join((self.__module__, self.__class__.__name__))
            + "[%s]" % self.uuid
        )

    def __init__(self, work_directory, classifier, work_ingest, uid=None):
        """ Initialize IQR session

        :param work_directory: Directory we are allowed to use for working files
        :type work_directory: str

        :param classifier: Classifier to use for this IQR session
        :type classifier: SMQTK.Classifiers.SMQTKClassifier

        :param work_ingest: Ingest to add extension files to
        :type work_ingest: SMQTK.utils.DataIngest.DataIngest

        :param uid: Optional manual specification of session UUID.
        :type uid: str or uuid.UUID

        """
        self.lock = multiprocessing.RLock()
        self.uuid = uuid.uuid1() if uid is None else uid

        self.positive_ids = set()
        self.negative_ids = set()

        self._work_dir = work_directory

        self._classifier = classifier

        # Mapping of a clip ID to the probability of it being associated to
        # positive adjudications. This is None before any refinement occurs.
        #: :type: None or dict of (int, float)
        self.results = None

        # Ingest where extension images are placed
        self.extension_ingest = work_ingest

    def __del__(self):
        # Clean up working directory
        shutil.rmtree(self.work_dir)

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    @property
    def work_dir(self):
        if not osp.isdir(self._work_dir):
            safe_create_dir(self._work_dir)
        return self._work_dir

    @property
    def ordered_results(self):
        """
        Return a tuple of the current (id, probability) result pairs in
        order of probability score. If there are no results yet, None is
        returned.

        """
        with self.lock:
            if self.results:
                return tuple(sorted(self.results.iteritems(),
                                    key=lambda p: p[1],
                                    reverse=True))
            return None

    def extend_model(self, *image_files):
        """
        Extend our data models given the following image file paths.

        Given image files are added to this session's extension ingest.

        :raises ValueError: If an image file is already ingested.

        :param image_files: Iterable of image file paths
        :type image_files: Iterable of str

        """
        # with self.lock:
        #     p_pool = multiprocessing.pool.Pool()
        #
        #     args = []
        #     for img in image_files:
        #         uid, md5, fpath = self.extension_ingest.ingest_image(img)
        #         args.append((self._log.name, self.descriptor, uid, fpath))
        #
        #     self._log.info("Feature generation...")
        #     img_features = \
        #         p_pool.map_async(_iqr_async_image_feature, args).get()
        #
        #     p_pool.close()
        #     p_pool.join()
        #
        #     self._log.info("Updating FM")
        #     new_ids = []
        #     for img_id, img, feat in img_features:
        #         self._log.info("=== %s", img)
        #         # TODO: Update this function in FeatureMemory to take multiple
        #         #       ID, feature pairs (or parallel arrays)
        #         self.feature_memory.update(img_id, feat)
        #         new_ids.append(img_id)
        #
        #     # adding new IDs to positive adjudications set
        #     self.positive_ids.update(new_ids)

    def adjudicate(self, new_positives=(), new_negatives=(),
                   un_positives=(), un_negatives=()):
        """
        Update current state of user defined positive and negative truths on
        specific image IDs

        :param new_positives: New IDs of items to now be considered positive.
        :type new_positives: tuple of int
        :param new_negatives: New IDs of items to now be considered negative.
        :param un_positives: New item IDs that are now not positive any more.
        :type un_positives: tuple of int
        :param un_negatives: New item IDs that are now not negative any more.
        :type un_negatives: tuple of int

        """
        with self.lock:
            self.positive_ids.update(new_positives)
            self.positive_ids.difference_update(un_positives)
            self.positive_ids.difference_update(new_negatives)

            self.negative_ids.update(new_negatives)
            self.negative_ids.difference_update(un_negatives)
            self.negative_ids.difference_update(new_positives)

            # # EXPERIMENT
            # # When we have negative adjudications, remove use of the original
            # # bg IDs set in the feature memory, injecting this session's
            # # negative ID set (all both use set objects, so just share the ptr)
            # # When we don't have negative adjudications, reinstate the original
            # # set of bg IDs.
            # if self.negative_ids:
            #     self.feature_memory._bg_clip_ids = self.negative_ids
            # else:
            #     self.feature_memory._bg_clip_ids = self._original_fm_bgid_set

            # # Update background flags in our feature_memory
            # # - new positives and un-negatives are now non-background
            # # - new negatives are now background.
            # for uid in set(new_positives).union(un_negatives):
            #     self._log.info("Marking UID %d as non-background", uid)
            #     self.feature_memory.update(uid, is_background=False)
            #     assert uid not in self.feature_memory.get_bg_ids()
            # for uid in new_negatives:
            #     self._log.info("Marking UID %d as background", uid)
            #     self.feature_memory.update(uid, is_background=True)
            #     assert uid in self.feature_memory.get_bg_ids()

    def refine(self, new_positives=(), new_negatives=(),
               un_positives=(), un_negatives=()):
        """ Refine current model results based on current adjudication state

        :raises RuntimeError: There are no adjudications to run on. We must have
            at least one positive adjudication.

        :param new_positives: New IDs of items to now be considered positive.
        :type new_positives: tuple of int
        :param new_negatives: New IDs of items to now be considered negative.
        :param un_positives: New item IDs that are now not positive any more.
        :type un_positives: tuple of int
        :param un_negatives: New item IDs that are now not negative any more.
        :type un_negatives: tuple of int

        """
        # with self.lock:
        #     self.adjudicate(new_positives, new_negatives, un_positives,
        #                     un_negatives)
        #
        #     if not self.positive_ids:
        #         raise RuntimeError("Did not find at least one positive "
        #                            "adjudication.")
        #
        #     #
        #     # Model training
        #     #
        #     self._log.info("Starting model training...")
        #     self._log.debug("-- Positives: %s", self.positive_ids)
        #     self._log.debug("-- Negatives: %s", self.negative_ids)
        #
        #     # query submatrix of distance kernel for positive and background
        #     # IDs.
        #     self._log.debug("Extracting symmetric submatrix")
        #     idx2id_map, idx_bg_flags, m = \
        #         self.feature_memory\
        #             .get_distance_kernel()\
        #             .symmetric_submatrix(*self.positive_ids)
        #     self._log.debug("-- num bg: %d", idx_bg_flags.count(True))
        #     self._log.debug("-- m shape: %s", m.shape)
        #
        #     # for model training function, inverse of idx_is_bg: True
        #     # indicates a positively adjudicated index
        #     labels_train = numpy.array(tuple(not b for b in idx_bg_flags))
        #
        #     # # Where to save working models
        #     # model_filepath = osp.join(self.work_dir,
        #     #                           "iqr_session.%s.model" % self.uuid)
        #     # svIDs_filepath = osp.join(self.work_dir,
        #     #                           "iqr_session.%s.svIDs" % self.uuid)
        #
        #     # Returned dictionary contains the keys "model" and "clipid_SVs"
        #     # referring to the trained model and a list of support vectors,
        #     # respectively.
        #     ret_dict = iqr_model_train(m, labels_train, idx2id_map,
        #                                self.svm_train_params)
        #     svm_model = ret_dict['model']
        #     svm_svIDs = ret_dict['clipids_SVs']
        #
        #     #
        #     # Model Testing/Application
        #     #
        #     self._log.info("Starting model application...")
        #
        #     # As we're extracting rows, the sum of IDs are preserved along
        #     # the x-axis (column IDs). The list of IDs along the x-axis is
        #     # then effectively the ordered list of all IDs
        #     idx2id_row, idx2id_col, kernel_test = \
        #         self.feature_memory.get_distance_kernel()\
        #                            .extract_rows(svm_svIDs)
        #
        #     # Testing/Ranking call
        #     #   Passing the array version of the kernel sub-matrix. The
        #     #   returned output['probs'] type matches the type passed in
        #     #   here, and using an array makes syntax cleaner.
        #     self._log.debug("Ranking IDs")
        #     output = iqr_model_test(svm_model, kernel_test.A, idx2id_col)
        #
        #     probability_map = dict(zip(output['clipids'], output['probs']))
        #     if self.results is None:
        #         self.results = IqrResultsDict()
        #     self.results.update(probability_map)
        #
        #     # Force adjudicated negatives to be probability 0.0 since we don't
        #     # want them possibly polluting the further adjudication views.
        #     for uid in self.negative_ids:
        #         self.results[uid] = 0.0

    def reset(self):
        """ Reset the IQR Search state

        No positive adjudications, reload original feature data

        """
        # with self.lock:
        #     self.positive_ids.clear()
        #     self.negative_ids.clear()
        #     # noinspection PyUnresolvedReferences
        #     self.feature_memory = FeatureMemory.construct_from_files(
        #         self.descriptor.ids_file, self.descriptor.bg_flags_file,
        #         self.descriptor.feature_data_file,
        #         self.descriptor.kernel_data_file
        #     )
        #     self.results = None
        #
        #     # clear contents of working directory
        #     shutil.rmtree(self.work_dir)
        #     os.makedirs(self.work_dir)
