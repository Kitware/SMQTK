"""
LICENCE
-------
Copyright 2013-2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

from collections import deque
import multiprocessing
from threading import current_thread
import time


class ReaderUpdateException (Exception):
    """
    Exception thrown when an acquireWrite() call is called within a reader lock
    while another reader lock is already upgraded.
    """
    pass


# noinspection PyPep8Naming
class DummyRWLock (object):
    """
    Dummy object that mimics the API of a ReadWrite lock but doesn't actually
    do anything.
    """

    def acquireRead(self, timeout=None, _id=None):
        pass

    def acquireWrite(self, timeout=None, _id=None):
        pass

    def releaseRead(self, _id=None):
        pass

    def releaseWrite(self, _id=None):
        pass

    # noinspection PyMethodMayBeStatic
    def read_lock(self, timeout=None, _id=None):
        # noinspection PyMethodParameters
        class DummyReadWithLock (object):
            def __enter__(_self):
                pass
            def __exit__(_self, *args):
                pass
        return DummyReadWithLock()

    # noinspection PyMethodMayBeStatic
    def write_lock(self, timneout=None, _id=None):
        # noinspection PyMethodParameters
        class DummyWriteWithLock (object):
            def __enter__(_self):
                pass
            def __exit__(_self, *args):
                pass
        return DummyWriteWithLock()


# noinspection PyPep8Naming
class ReadWriteLock (object):
    """ Reentrant Read-Write for multiprocessing

    Allows multiple threads/processes to simultaneously hold a read lock, while
    allowing only a single thread/process to hold a write lock at the same point
    of time.

    WARNING: This lock is NOT write-lock re-entrant. Attempting to acquire a
    write lock within a write lock WILL cause a dead-lock.

    When a read lock is requested while a write lock is held, the reader
    is blocked; when a write lock is requested while another write lock is
    held or there are read locks, the writer is blocked.

    When resolving who acquires the lock between readers and writers, writers
    are always preferred by this implementation: if there are blocked
    threads waiting for a write lock, current readers may request more read
    locks (which they eventually should free, as they starve the waiting
    writers otherwise), but a new thread requesting a read lock will not
    be granted one, and block. This might mean starvation for readers if
    two writer threads interweave their calls to acquireWrite() without
    leaving a window only for readers.

    NOT IMPLEMENTED :: TODO ::
    In case a current reader requests a write lock, this can and will be
    satisfied without giving up the read locks first, but, only one thread
    may perform this kind of lock upgrade, as a deadlock would otherwise
    occur. After the write lock has been granted, the thread will hold a
    full write lock, and not be downgraded after the upgrading call to
    acquireWrite() has been match by a corresponding release().
        - could add a acquireWriteUpgrade method for the specific purpose of
          acquiring a write lock in the presence of readers.

    """

    def __init__(self):
        """ Initialize lock state
        """
        # NOTE: If ever we want an equal access priority instead of giving
        # access priority to the writers, we could have a global queue FIFO
        # queue instead of just the pending writers queue. This would move this
        # from a case-2 RWLock problem to a case-3, ensuring zero starvation of
        # either lock.

        # Condition lock on internal variables
        # noinspection PyUnresolvedReferences
        self.__cond = multiprocessing.Condition()

        # Records the ID of the current writer. If this is None, there isn't
        # currently and entity that has the write lock.
        #: :type: None or int
        self.__writer = None

        # Allows write locks to be reentrant. The write lock is only released
        # when this hits zero during a releaseWrite
        self.__writer_count = 0

        # Writers will be serviced before readers, recorded by this count.
        # Writers serviced in order of request as ordered in this list.
        #: :type: deque of (int, int)
        self.__pending_writers = deque()  # insert linked-list structure here?

        # current processes holding reader locks. Maps the user's ID to the
        # number of times it has reentered reading locks.
        #: :type: dict of (int, int)
        self.__readers = {}

    def acquireRead(self, timeout=None, _id=None):
        """ Acquire a read lock, waiting at most timeout seconds.

        If a timeout is not given, we will block indefinitely. If a negative
        timeout is given, this acquire will act as non-blocking. Raises
        RuntimeError if the timeout expires before lock acquired.

        NOTE: NO NOT give a value to ``_id`` unless you know what you're doing.
        Otherwise, it should be a tuple of two ints uniquely identifying the
        acquirer of the lock. Subsequent acquisition of locks by the same
        acquirer must provide the same value for correct functionality.
        Otherwise, when _id is None, this is the combination of
        (processID, threadID).

        :param timeout: Optional timeout on the lock acquire time in seconds.
        :type timeout: int or None

        """
        me = _id or (multiprocessing.current_process().ident,
                     current_thread().ident)
        # print("[DEBUG] acquireRead ID:", me)
        if timeout is not None:
            expire_time = time.time() + timeout
        with self.__cond:
            # If we're the writer, we should also be able to read since reading
            # from other sources is currently locked while we're the writer.
            # Increment our reader reentrant level. Required a matching
            # releaseRead within acquireWrite block.
            if self.__writer == me:
                self.__readers[me] = self.__readers.get(me, 0) + 1
                return
            while True:
                # Only consider granting a read lock if there is currently no
                # writer.
                if not self.__writer:
                    # Increment the reentrant level if we already have a read
                    # lock (including if we're an upgraded reader), else
                    # grant a read lock if there are no pending writers or an
                    # upgraded reader.
                    if self.__readers.get(me):
                        self.__readers[me] += 1
                        return
                    elif not self.__pending_writers:
                        self.__readers[me] = 1
                        return

                if timeout is not None:
                    # noinspection PyUnboundLocalVariable
                    remaining = expire_time - time.time()
                    if remaining <= 0:
                        raise RuntimeError("Timeout expired while waiting for "
                                           "read lock acquire.")
                    self.__cond.wait(remaining)
                else:
                    self.__cond.wait()

    def acquireWrite(self, timeout=None, _id=None):
        """ Acquire a write lock, waiting at most timeout seconds.

        If a timeout is not given, we will block indefinitely. If a negative
        timeout is given, this acquire will act as non-blocking. Raises
        RuntimeError if the timeout expires before lock acquired.

        NOTE: NO NOT give a value to ``_id`` unless you know what you're doing.
        Otherwise, it should be a tuple of two ints uniquely identifying the
        acquirer of the lock. Subsequent acquisition of locks by the same
        acquirer must provide the same value for correct functionality.
        Otherwise, when _id is None, this is the combination of
        (processID, threadID).

        :param timeout: Optional timeout on the lock acquire time in seconds.
        :type timeout: int or None

        """
        me = _id or (multiprocessing.current_process().ident,
                     current_thread().ident)
        # print("[DEBUG] acquireWrite ID:", me)
        if timeout is not None:
            expire_time = time.time() + timeout
        with self.__cond:
            # if we're either the writer or an upgraded reader already,
            # increment reentrant level and grant lock.
            if self.__writer == me:
                self.__writer_count += 1
                return

            # Notifying of no read lock upgrade ability at this time.
            elif self.__readers.get(me, False):
                raise ReaderUpdateException("Read lock upgrades not supported "
                                            "at this time.")

            # we're now a normal "pending" writer, no readers will acquire while
            # we are pending.
            else:
                self.__pending_writers.append(me)

            while True:
                # If no readers and no writer, we have clear passage. An
                # upgraded reader would have an entry in __readers if it
                # existed.
                if self.__writer is None and not self.__readers:
                    if self.__pending_writers[0] == me:
                        assert (self.__writer is None and
                                self.__writer_count == 0)
                        self.__writer = self.__pending_writers.popleft()
                        self.__writer_count = 1
                        return

                if timeout is not None:
                    # noinspection PyUnboundLocalVariable
                    remaining = expire_time - time.time()
                    if remaining <= 0:
                        # Timeout expired. Pull ourself from pending deque.
                        # (must be in the queue since in order to get here we
                        # had to have added ourself above).
                        self.__pending_writers.remove(me)
                        raise RuntimeError("Timeout expired while waiting for "
                                           "write lock acquire.")
                    self.__cond.wait(remaining)
                else:
                    self.__cond.wait()

    def releaseRead(self, _id=None):
        """ Release currently acquired read lock.

        Raises RuntimeError if we try to release a read lock when there isn't
        one acquired.

        NOTE: NO NOT give a value to ``_id`` unless you know what you're doing.
        Otherwise, it should be a tuple of two ints uniquely identifying the
        acquirer of the lock. Subsequent acquisition of locks by the same
        acquirer must provide the same value for correct functionality.
        Otherwise, when _id is None, this is the combination of
        (processID, threadID).

        """
        me = _id or (multiprocessing.current_process().ident,
                     current_thread().ident)
        # print("[DEBUG] releaseRead ID:", me)
        with self.__cond:
            # Releasing an inner read lock within an outer write lock. Must have
            # at least one reader level available, else this is an unbound
            # release (no paired acquire).
            if self.__writer == me:
                level = self.__readers.get(me, 0)
                if level > 0:
                    self.__readers[me] -= 1
                    # When out of reader levels, remove us from the map
                    if not self.__readers[me]:
                        del self.__readers[me]
                elif level < 0:
                    raise RuntimeError("Achieved negative reader reentrant "
                                       "level. There's a bug somewhere.")
                else:
                    raise RuntimeError("Attempted a read lock release within "
                                       "a write lock block when no read lock "
                                       "was acquired.")

            # if we're in a read lock
            elif self.__readers.get(me, None):
                # decrement reader count, notify
                self.__readers[me] -= 1

                # If we've released our last read lock, remove ourselves from
                # the mapping of active readers.
                if not self.__readers[me]:
                    del self.__readers[me]

                    # If there are no more readers left, notify-all for those
                    # waiting on this condition.
                    if not self.__readers:
                        self.__cond.notify_all()

            # Erroneous release
            else:
                raise RuntimeError("Attempting release of read lock when one "
                                   "not acquired.")

    def releaseWrite(self, _id=None):
        """ Release currently acquired write lock.

        Raises RuntimeError if we try to release a write lock when there isn't
        one acquired.

        NOTE: NO NOT give a value to ``_id`` unless you know what you're doing.
        Otherwise, it should be a tuple of two ints uniquely identifying the
        acquirer of the lock. Subsequent acquisition of locks by the same
        acquirer must provide the same value for correct functionality.
        Otherwise, when _id is None, this is the combination of
        (processID, threadID).

        """
        me = _id or (multiprocessing.current_process().ident,
                     current_thread().ident)
        # print("[DEBUG] releaseWrite ID:", me)
        with self.__cond:
            # Obviously, only decrement when we own the write lock
            if self.__writer == me:
                self.__writer_count -= 1

                # when we release our initial entry level, clear us from the
                # writer slot
                if not self.__writer_count:
                    # falsify writer flag, notify
                    self.__writer = None
                    self.__cond.notify_all()

            # Erroneous release
            else:
                raise RuntimeError("Attempting release of write lock when one "
                                   "not acquired.")

    def read_lock(self, timeout=None, _id=None):
        """
        Return an object to be used for ``with`` statements that acquires and
        releases read locks. A timeout may be specified, which will be used when
        attempting to acquire our lock.

        NOTE: NO NOT give a value to ``_id`` unless you know what you're doing.
        Otherwise, it should be a tuple of two ints uniquely identifying the
        acquirer of the lock. Subsequent acquisition of locks by the same
        acquirer must provide the same value for correct functionality.
        Otherwise, when _id is None, this is the combination of
        (processID, threadID).

        :param timeout: Optional timeout on the lock acquire in seconds.
        :type timeout: int

        :return: New object instance with __enter__ and __exit__ methods
            defined.
        :rtype: object

        """
        me = _id or (multiprocessing.current_process().ident,
                     current_thread().ident)

        # noinspection PyMethodParameters,PyUnusedLocal
        class ReadWithLock (object):

            def __enter__(_self):
                self.acquireRead(timeout, me)

            def __exit__(_self, *args):
                self.releaseRead(me)

        return ReadWithLock()

    def write_lock(self, timeout=None, _id=None):
        """
        Return an object to be used for ``with`` statements that acquires and
        releases write locks. A timeout may be specified, which will be used
        when attempting to acquire our lock.

        NOTE: NO NOT give a value to ``_id`` unless you know what you're doing.
        Otherwise, it should be a tuple of two ints uniquely identifying the
        acquirer of the lock. Subsequent acquisition of locks by the same
        acquirer must provide the same value for correct functionality.
        Otherwise, when _id is None, this is the combination of
        (processID, threadID).

        :param timeout: Optional timeout on the lock acquire in seconds.
        :type timeout: int

        :return: New object instance with __enter__ and __exit__ methods
            defined.
        :rtype: object

        """
        me = _id or (multiprocessing.current_process().ident,
                     current_thread().ident)

        # noinspection PyMethodParameters,PyUnusedLocal
        class WriteWithLock (object):

            def __enter__(_self):
                self.acquireWrite(timeout, me)

            def __exit__(_self, *args):
                self.releaseWrite(me)

        return WriteWithLock()
