# coding=utf-8
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import shutil
import uuid

from .iqr_session import IqrSession, IqrResultsDict


class IqrSessionFusion (IqrSession):
    """
    Encapsulation of IQR Session related data structures with a centralized lock
    for multi-thread access.

    This object is compatible with the python with-statement, so when elements
    are to be used or modified, it should be within a with-block so race
    conditions do not occur across threads/sub-processes.

    """

    def __init__(self, work_directory, reactor, work_ingest,
                 session_uid=None):
        """ Initialize IQR session

        :param work_directory: Directory we are allowed to use for working files
        :type work_directory: str

        :param reactor: fusion reactor to drive online extension and indexing
        :type reactor: smqtk.fusion.Reactor.Reactor

        :param work_ingest: Ingest to add extension files to
        :type work_ingest: SMQTK.utils.DataIngest.DataIngest

        :param session_uid: Optional manual specification of session UUID.
        :type session_uid: str or uuid.UUID

        """
        # noinspection PyTypeChecker
        super(IqrSessionFusion, self).__init__(work_directory, None, None,
                                               work_ingest, session_uid)

        self.reactor = reactor

    def refine(self, new_positives=(), new_negatives=(),
               un_positives=(), un_negatives=()):
        """ Refine current model results based on current adjudication state

        :raises RuntimeError: There are no adjudications to run on. We must have
            at least one positive adjudication.

        :param new_positives: New IDs of items to now be considered positive.
        :type new_positives: collections.Iterable of int

        :param new_negatives: New IDs of items to now be considered negative.
        :type new_negatives: collections.Iterable of int

        :param un_positives: New item IDs that are now not positive any more.
        :type un_positives: collections.Iterable of int

        :param un_negatives: New item IDs that are now not negative any more.
        :type un_negatives: collections.Iterable of int

        """
        with self.lock:
            self.adjudicate(new_positives, new_negatives, un_positives,
                            un_negatives)

            if not self.positive_ids:
                raise RuntimeError("Did not find at least one positive "
                                   "adjudication.")

            id_probability_map = \
                self.reactor.rank(self.positive_ids, self.negative_ids)

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
            self.reactor.reset()
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
