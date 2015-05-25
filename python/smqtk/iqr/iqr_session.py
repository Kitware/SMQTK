# coding=utf-8
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging
import multiprocessing
import multiprocessing.pool
import os.path as osp
import shutil
import uuid

from smqtk.utils import safe_create_dir


class IqrResultsDict (dict):
    """
    Dictionary subclass for standardizing data types stored.
    """

    def __setitem__(self, i, v):
        super(IqrResultsDict, self).__setitem__(i, float(v))

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
    """
    Encapsulation of IQR Session related data structures with a centralized lock
    for multi-thread access.

    This object is compatible with the python with-statement, so when elements
    are to be used or modified, it should be within a with-block so race
    conditions do not occur across threads/sub-processes.

    """

    @property
    def _log(self):
        return logging.getLogger(
            '.'.join((self.__module__, self.__class__.__name__))
            + "[%s]" % self.uuid
        )

    def __init__(self, work_directory, descriptor, indexer, work_ingest,
                 session_uid=None):
        """ Initialize IQR session

        :param work_directory: Directory we are allowed to use for working files
        :type work_directory: str

        :param descriptor: Descriptor to use for this IQR session
        :type descriptor: smqtk.content_description.ContentDescriptor

        :param indexer: indexer to use for this IQR session
        :type indexer: smqtk.indexing.Indexer

        :param work_ingest: Ingest to add extension files to
        :type work_ingest: smqtk.data_rep.DataSet

        :param session_uid: Optional manual specification of session UUID.
        :type session_uid: str or uuid.UUID

        """
        self.uuid = session_uid or uuid.uuid1()
        self.lock = multiprocessing.RLock()

        self.positive_ids = set()
        self.negative_ids = set()

        self._work_dir = work_directory

        self.descriptor = descriptor
        self.indexer = indexer

        # Mapping of a clip ID to the probability of it being associated to
        # positive adjudications. This is None before any refinement occurs.
        #: :type: None or dict of (collections.Hashable, float)
        self.results = None

        # Ingest where extension images are placed
        self.extension_ingest = work_ingest

    def __del__(self):
        # Clean up working directory
        if osp.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)

    def __enter__(self):
        """
        :rtype: IqrSession
        """
        self.lock.acquire()
        return self

    # noinspection PyUnusedLocal
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

    def adjudicate(self, new_positives=(), new_negatives=(),
                   un_positives=(), un_negatives=()):
        """
        Update current state of user defined positive and negative truths on
        specific image IDs

        :param new_positives: New IDs of items to now be considered positive.
        :type new_positives: collections.Iterable of collections.Hashable

        :param new_negatives: New IDs of items to now be considered negative.
        :type new_negatives: collections.Iterable of collections.Hashable

        :param un_positives: New item IDs that are now not positive any more.
        :type un_positives: collections.Iterable of collections.Hashable

        :param un_negatives: New item IDs that are now not negative any more.
        :type un_negatives: collections.Iterable of collections.Hashable

        """
        with self.lock:
            self.positive_ids.update(new_positives)
            self.positive_ids.difference_update(un_positives)
            self.positive_ids.difference_update(new_negatives)

            self.negative_ids.update(new_negatives)
            self.negative_ids.difference_update(un_negatives)
            self.negative_ids.difference_update(new_positives)

    def refine(self, new_positives=(), new_negatives=(),
               un_positives=(), un_negatives=()):
        """ Refine current model results based on current adjudication state

        :raises RuntimeError: There are no adjudications to run on. We must have
            at least one positive adjudication.

        :param new_positives: New IDs of items to now be considered positive.
        :type new_positives: collections.Iterable of collections.Hashable

        :param new_negatives: New IDs of items to now be considered negative.
        :type new_negatives: collections.Iterable of collections.Hashable

        :param un_positives: New item IDs that are now not positive any more.
        :type un_positives: collections.Iterable of collections.Hashable

        :param un_negatives: New item IDs that are now not negative any more.
        :type un_negatives: collections.Iterable of collections.Hashable

        """
        with self.lock:
            self.adjudicate(new_positives, new_negatives, un_positives,
                            un_negatives)

            if not self.positive_ids:
                raise RuntimeError("Did not find at least one positive "
                                   "adjudication.")

            id_probability_map = \
                self.indexer.rank_model(self.positive_ids, self.negative_ids)

            if self.results is None:
                self.results = IqrResultsDict()
            self.results.update(id_probability_map)

            # Force adjudicated positives and negatives to be probability 1 and
            # 0, respectively, since we want to control where they show up in
            # our results view.
            for uid in self.positive_ids:
                self.results[uid] = 1.0
            for uid in self.negative_ids:
                self.results[uid] = 0.0

    def reset(self):
        """ Reset the IQR Search state

        No positive adjudications, reload original feature data

        """
        with self.lock:
            self.positive_ids.clear()
            self.negative_ids.clear()
            self.indexer.reset()
            self.results = None

            # clear contents of working directory
            shutil.rmtree(self.work_dir)

            # Re-initialize extension ingest. Now that we're killed the IQR work
            # tree, this should initialize empty
            if len(self.extension_ingest):
                #: :type: smqtk.utils.DataIngest.DataIngest
                self.extension_ingest = self.extension_ingest.__class__(
                    self.extension_ingest.data_directory,
                    self.extension_ingest.work_directory,
                    starting_index=min(self.extension_ingest.uids())
                )
