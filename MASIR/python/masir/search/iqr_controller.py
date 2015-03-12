# coding=utf-8

import multiprocessing

from masir.search import IqrSession
from masir.search.colordescriptor import ColorDescriptor_CSIFT

import masir_config


class IqrController (object):
    """
    Main controlling object for one or more IQR Sessions.

    In order to interface with a web server, methods defined here are
    non-blocking (except for thread contention) and thread-safe.

    This class may be used with the ``with`` statement. This will enable the
    instance's primary lock, preventing any other action from being performed on
    the instance while inside the with statement. The lock is reentrant, so
    nested with-statements will not dead-lock.

    """

    def __init__(self):
        # Map of uuid to the search state
        #: :type: dict of (uuid.UUID, IqrSession)
        self._iqr_sessions = {}
        # RLock for each iqr session. managed in parallel to iqr_session map
        #: :type: dict of (uuid.UUID, RLock)
        self._iqr_sessions_locks = {}

        # RLock for iqr_session{_locks} maps.
        self._map_rlock = multiprocessing.RLock()

    def __enter__(self):
        self._map_rlock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._map_rlock.release()

    def session_uuids(self):
        """
        Return a tuple of all currently registered IqrSessions.

        :return: a tuple of all currently registered IqrSessions.
        :rtype: tuple of uuid.UUID

        """
        with self._map_rlock:
            return tuple(self._iqr_sessions)

    def has_session_uuid(self, session_uuid):
        """ Check if this controller contains a session referenced by the given
        ID.

        Performance using this function is faster compared to getting all UUIDs
        and performing a linear search.

        :param session_uuid: Possible UUID of a session
        :type session_uuid: uuid.UUID

        :return: True of the given UUID references a session in this controller
            and false if not.
        :rtype: bool

        """
        with self._map_rlock:
            return session_uuid in self._iqr_sessions

    def with_session(self, session_uuid):
        """
        Return a context object to allow use of a session while access locked.
        This prevents removal while being used.

        :param session_uuid: UUID of the session to get
        :type session_uuid: uuid.UUID

        :raises KeyError: There is no session associated with the given UUID
            (may have never been or removed already).

        :return: Return an object to use with the python with-statement to allow
            use of the requested session within a protection lock.
        :rtype: _session_context_

        """
        with self._map_rlock:
            if session_uuid not in self._iqr_sessions:
                raise KeyError(session_uuid)

            with self._iqr_sessions_locks[session_uuid]:

                # noinspection PyMethodParameters,PyPep8Naming
                class _session_context_ (object):
                    def __init__(inner_self):
                        inner_self.sess_lock = self._iqr_sessions_locks[session_uuid]
                        inner_self.sess = self._iqr_sessions[session_uuid]

                    def __enter__(inner_self):
                        """
                        :rtype: IqrSession
                        """
                        inner_self.sess_lock.acquire()
                        return inner_self.sess

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        self.sess_lock.release()

                return _session_context_()

    def init_new_session(self, descriptor):
        """ Initialize a new IQR Session, returning the uuid of that session

        :param descriptor: The descriptor to focus on for this IQR session
        :type descriptor: masir.search.FeatureDescriptor.FeatureDescriptor

        :return: UUID of new IQR Session
        :rtype: uuid.UUID

        """
        with self._map_rlock:
            new_session = IqrSession(masir_config.DIR_WORK, descriptor)
            self._iqr_sessions[new_session.uuid] = new_session
            self._iqr_sessions_locks[new_session.uuid] = multiprocessing.RLock()
            return new_session.uuid

    def remove_session(self, session_uuid):
        """
        Remove an IQR Session by session UUID

        If no IQR Session exists for the given UUID, we return false but
        otherwise do nothing

        :param session_uuid: Session UUID
        :type session_uuid: uuid.UUID

        :return: True if a session was removed, False if not.
        :rtype: bool

        """
        with self._map_rlock:
            if session_uuid in self._iqr_sessions:
                with self._iqr_sessions_locks[session_uuid]:
                    del self._iqr_sessions[session_uuid]
                    del self._iqr_sessions_locks[session_uuid]
                    return True
            else:
                return False
