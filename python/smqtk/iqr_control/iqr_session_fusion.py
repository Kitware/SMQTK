# coding=utf-8
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import uuid

from .iqr_session import IqrSession


class IqrSessionFusion (IqrSession):
    """
    Encapsulation of IQR Session related data structures with a centralized lock
    for multi-thread access.

    This object is compatible with the python with-statement, so when elements
    are to be used or modified, it should be within a with-block so race
    conditions do not occur across threads/sub-processes.

    """

    def __init__(self, work_directory, reactor, session_uid=None):
        """ Initialize IQR session

        :param work_directory: Directory we are allowed to use for working files
        :type work_directory: str

        :param reactor: fusion reactor to drive online extension and indexing
        :type reactor: smqtk.fusion.reactor.Reactor

        :param session_uid: Optional manual specification of session UUID.
        :type session_uid: str or uuid.UUID

        """
        # noinspection PyTypeChecker
        super(IqrSessionFusion, self).__init__(work_directory, None, None,
                                               session_uid)

        self.indexer = reactor
