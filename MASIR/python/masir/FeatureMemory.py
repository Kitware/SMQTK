# coding=utf-8
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

# -*- coding: utf-8 -*-

from _abcoll import Iterable
from collections import deque
import logging
from multiprocessing import current_process
from multiprocessing.managers import (
    all_methods,
    BaseProxy,
    State,
    SyncManager
)
import multiprocessing
import multiprocessing.util

import numpy as np
from numpy.core.multiarray import ndarray
# Where the matrix class is depends on version. Either in numpy.core.defmatrix
# or in numpy.matrixlib.defmatrix.
try:
    # noinspection PyUnresolvedReferences
    from numpy.matrixlib.defmatrix import matrix
except ImportError:
    # noinspection PyUnresolvedReferences
    from numpy.core.defmatrix import matrix
    # If this doesn't import, we're either missing version support for the
    # installed version of numpy

from threading import current_thread
import time

from .SimpleTimer import SimpleTimer


class ReaderUpdateException (Exception):
    """
    Exception thrown when an acquireWrite() call is called within a reader lock
    while another reader lock is already upgraded.
    """
    pass


# noinspection PyPep8Naming
class DummyRWLock (object):

    def acquireRead(self, timeout=None, _id=None):
        pass

    def acquireWrite(self, timeout=None, _id=None):
        pass

    def releaseRead(self, _id=None):
        pass

    def releaseWrite(self, _id=None):
        pass

    def read_lock(self, timeout=None, _id=None):
        # noinspection PyMethodParameters
        class DummyReadWithLock (object):
            def __enter__(_self):
                pass
            def __exit__(_self, *args):
                pass
        return DummyReadWithLock()

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
        me = _id or (current_process().ident, current_thread().ident)
        # print "[DEBUG] acquireRead ID:", me
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
        me = _id or (current_process().ident, current_thread().ident)
        # print "[DEBUG] acquireWrite ID:", me
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
        me = _id or (current_process().ident, current_thread().ident)
        # print "[DEBUG] releaseRead ID:", me
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
        me = _id or (current_process().ident, current_thread().ident)
        # print "[DEBUG] releaseWrite ID:", me
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
        me = _id or (current_process().ident, current_thread().ident)

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
        me = _id or (current_process().ident, current_thread().ident)

        # noinspection PyMethodParameters,PyUnusedLocal
        class WriteWithLock (object):

            def __enter__(_self):
                self.acquireWrite(timeout, me)

            def __exit__(_self, *args):
                self.releaseWrite(me)

        return WriteWithLock()


class DistanceKernel (object):
    """
    Feature Distance Kernel object.

    This class allows the kernel to either be symmetric or not. If it is
    symmetric, the ``symmetric_submatrix`` function becomes available.

    Intended to be used with FeatureManager proxy objects (given at
    construction)

    MONKEY PATCHING:
    When using this object directly (not using the FeatureManager stuff) and
    sending it over pipes, the ReadWriteLock needs to be monkey patched out (the
    multiprocessing.Condition variable doesn't play nicely). Need to set an
    instance of a DummyRWLock to the DistanceKernel._rw_lock property. For
    example:

        ...
        dk = ...
        dk._rw_lock = DummyRWLock()
        <send dk into a pipe>
        ...

    """

    @classmethod
    def construct_symmetric_from_files(cls, id_vector_file, kernel_mat_file,
                                       bg_flags_file=None):
        """
        Construct a symmetric DistanceKernel object, requiring a background
        flags file to denote clip IDs that are to be treated as background
        clips (required to activate symmetric_submatrix function). Such a

        DistanceKernel is usually used with event learning and should be
        provided a background flags file also.

        :param id_vector_file: File containing the numpy.savetxt(...) output of
            clip ID values in the order in which they associate to the rows of
            the kernel matrix.
        :type id_vector_file: str
        :param kernel_mat_file: File containing the kernel matrix as saved by
            numpy.save(...) (saved as an ndarray, converted to matrix on load).
        :type kernel_mat_file: str
        :param bg_flags_file: Optional file containing output of
            numpy.savetxt(...) where each index maps a row index of the kernel
            to whether or not the associated clip ID should be considered a
            background video or not.
        :type bg_flags_file: str
        :return: Symmetric DistanceKernel constructed with the data provided in
            the provided files.
        :rtype: DistanceKernel

        """
        clip_ids = np.array(np.loadtxt(id_vector_file))
        # noinspection PyCallingNonCallable
        kernel_mat = np.matrix(np.load(kernel_mat_file))

        if bg_flags_file is not None:
            bg_flags = np.array(np.loadtxt(bg_flags_file))
            bg_clips = set([clip_ids[i]
                            for i, e in enumerate(bg_flags)
                            if e])
        else:
            bg_clips = None

        return DistanceKernel(clip_ids, clip_ids, kernel_mat, bg_clips)

    @classmethod
    def construct_asymmetric_from_files(cls, row_ids_file, col_ids_file,
                                        kernel_mat_file):
        """
        Construct an asymmetric DistanceKernel object, usually used for archive
        searches.

        No option for providing background clip IDs as asymmetric kernels are
        NOT used for learning purposes.

        :param row_ids_file: File containing the numpy.savetxt(...) output of
            clip ID values in the order in which they associate to the rows of
            the given kernel matrix.
        :type row_ids_file: str
        :param col_ids_file: File containing the numpy.savetxt(...) output of
            clip ID values in the order in which they associate to the columns
            of the given kernel matrix.
        :type col_ids_file: str
        :param kernel_mat_file: File containing the kernel matrix as saved by
            numpy.save(...) (saved as an ndarray, converted to matrix on load).
        :type kernel_mat_file: str
        :return: Asymmetric DistanceKernel constructed with the data provided in
            the provided files.
        :rtype: DistanceKernel

        """
        row_cids = np.array(np.loadtxt(row_ids_file))
        col_cids = np.array(np.loadtxt(col_ids_file))
        # noinspection PyCallingNonCallable
        kernel_mat = np.matrix(np.load(kernel_mat_file))
        return DistanceKernel(row_cids, col_cids, kernel_mat)

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __init__(self, row_id_index_map, col_id_index_map, kernel_mat,
                 bg_clip_ids=None, rw_lock=None):
        """
        Initialize the kernel matrix. The initialization values will more than
        likely be proxies to np.matrix objects.

        The ``bg_clip_ids`` array may be given when this kernel matrix is to be
        a square, symmetric kernel and activates the use of the
        ``symmetric_submatrix`` method. This array must list clip IDs that are
        to be considered "background" IDs, or clips that are to always be
        considered negative. These clip IDs must be included in symmetric
        sub-matrices.

        This array must be the same dimension as
        the row and column indices, containing boolean flags. These flags mark
        that the clip ID found at the same index in the edge ID maps should be
        considered a "background" ID, or one that is always treated as a
        negative. This is for the stipulation in the symmetric_submatrix method
        that it always includes the background ID set in the submatrix.

        :param row_id_index_map: Array of clip IDs associated to row indices.
            Contents will be treated as ints.
        :type row_id_index_map: ndarray of int
        :param col_id_index_map: Array of clip IDs associated to row indices.
            Contents will be treated as ints.
        :type col_id_index_map: ndarray of int
        :param kernel_mat: Kernel data matrix.
        :type kernel_mat: matrix
        :param bg_clip_ids: Optional array of integers, marking whether a
            clip ID should be considered a "background" video. Contents will be
            treated as ints.
        :type bg_clip_ids: set of int
        :param rw_lock: Read-Write lock for data provided. This should be
            provided if the any of the data is shared with other objects/
            sources. If this is given None (default), then a lock is created.
        :type rw_lock: ReadWriteLock or None

        """
        # TODO: Possibly add checks for the id arrays like there is for the
        #       bgclipid array (int-able contents)
        assert np.shape(row_id_index_map)[0] == np.shape(kernel_mat)[0], \
            "Length of row index map and kernel row count did not match! " \
            "(row index map: %d, kernel row count: %d)" \
            % (np.shape(row_id_index_map)[0], np.shape(kernel_mat)[0])
        assert np.shape(col_id_index_map)[0] == np.shape(kernel_mat)[1], \
            "Length of col index map and kernel col count did not match! " \
            "(col index map: %d, kernel col count: %d)" \
            % (np.shape(col_id_index_map)[0], np.shape(kernel_mat)[1])

        self._row_id_index_map = row_id_index_map
        self._col_id_index_map = col_id_index_map
        self._kernel = kernel_mat

        assert ((bg_clip_ids is None)
                or isinstance(bg_clip_ids, (set, frozenset))), \
            "Must either given None or a set for the bg_clip_ids " \
            "vector. Got: %s" % type(bg_clip_ids)
        self._bg_cid_set = bg_clip_ids
        if bg_clip_ids is not None:
            try:
                [int(e) for e in bg_clip_ids]
            except Exception:
                raise ValueError("Not all of the contents of of bg_clip_ids "
                                 "could be treated as ints!")

        if rw_lock:
            assert isinstance(rw_lock, ReadWriteLock), \
                "Did not receive valid istance of RW Lock. Got '%s'" \
                % type(rw_lock)
            self._rw_lock = rw_lock
        else:
            self._rw_lock = ReadWriteLock()

    def get_lock(self):
        """
        :return: This object's read/write lock.
        :rtype: ReadWriteLock
        """
        return self._rw_lock

    def row_id_map(self):
        """
        :return: Row index-to-clipID map
        :rtype: ndarray
        """
        with self.get_lock().read_lock():
            return self._row_id_index_map

    def col_id_map(self):
        """
        :return: Column index-to-clipID map
        :rtype: ndarray
        """
        with self.get_lock().read_lock():
            return self._col_id_index_map

    def get_kernel_matrix(self):
        """
        RETURNED OBJECTS NOT THREAD/PROCESS SAFE. Once retrieved, if
        matrix may be modified by another thread/process

        :return: The underlying kernel matrix.
        :rtype: matrix

        """
        with self.get_lock().read_lock():
            return self._kernel

    def get_background_ids(self):
        """
        RETURNED OBJECTS NOT THREAD/PROCESS SAFE

        :return: The set of background clip IDs. May be None if there was no
            background set initialized.
        :rtype: None or frozenset

        """
        with self.get_lock().read_lock():
            return frozenset(self._bg_cid_set) \
                if self._bg_cid_set is not None \
                else frozenset()

    def is_symmetric(self):
        """
        :return: True if this is a square kernel matrix. This means that clip
            IDs along the row and column axes are the same and in the same order
            (starting from [0,0] and moving outwards).
        :rtype: bool

        """
        with self._rw_lock.read_lock():
            # Doing shape equality short circuit because the return value of
            # numpy.array equality changes depending on this condition, meaning
            # the use of the ...all() member function on the result is not
            # universally possible (i.e. when it returns a bool value when
            # shapes are not equal).

            # noinspection PyUnresolvedReferences
            return (self._row_id_index_map.shape == self._col_id_index_map.shape
                    and
                    (self._row_id_index_map == self._col_id_index_map).all())

    def symmetric_submatrix(self, *clip_ids):
        """
        Return a symmetric sub NxN matrix of the total distance kernel based on
        the clip IDs provided. The background clips will always be included in
        the matrix if this DistanceKernel was constructed with a list of
        background clip IDs.

        Clip IDs provided will be assumed non-background, or positive
        event examples. If the clip ID of a background video is provided as an
        argument, we will reconsider it as a non-background video in the
        returned index-to-is-background map (tuple).

        Note: The matrix returned will always be a new instance and not set up
        to use shared memory. When directly used with shared memory objects, it
        will be passed by value, not by reference.

        :param clip_ids: Integer clip IDs to include in the returned matrix. The
            returned matrix will contain all background clip IDs.
        :type clip_ids: Iterable of int
        :return: The index-to-clipID map (tuple), the index-to-is-background map
            (tuple) and the symmetric NxN submatrix, where N is the number of
            clip IDs provided as arguments plus the number of background IDs,
            minus the overlap between those two sets.
        :rtype: tuple of int, tuple of bool, matrix

        """
        with self._rw_lock.read_lock():
            with SimpleTimer("Checking inputs", self._log):
                if not self.is_symmetric():
                    raise RuntimeError("Cannot get a symmetric sub-matrix if "
                                       "the kernel is not square!")
                # DEPRECATED: Allowing the use of this method without explicitly
                #             providing background cIDs. This object will
                #             probably not ever be used this way, but there's no
                #             reason to explicitly disallow it.
                # if self._bg_cid_vec is None:
                #     raise RuntimeError("Cannot create the square submatrix "
                #                        "without the background flag vector!")

                try:
                    clip_ids = [int(e) for e in clip_ids]
                except:
                    raise ValueError("Not all clip IDs could be used as ints! "
                                     "Given: %s" % clip_ids)

                id_diff = set(clip_ids).difference(self._row_id_index_map)
                assert not id_diff, \
                    "Not all clip IDs provided are represented in this " \
                    "distance kernel matrix! (difference: %s)" \
                    % id_diff
                del id_diff

            with SimpleTimer("Computing union of BG clips and provided IDs",
                             self._log):
                if self._bg_cid_set is not None:
                    all_cids = self._bg_cid_set.union(clip_ids)
                else:
                    all_cids = set(clip_ids)

            # Reorder the given clip IDs so that they are in the same relative
            # order as the kernel matrix edges.
            focus_indices = []
            focus_clipids = []
            for idx, cid in enumerate(self._row_id_index_map):
                if (cid in all_cids) and (cid not in focus_clipids):
                    focus_indices.append(idx)
                    focus_clipids.append(cid)

            # index-to-isBG map for return
            # -> IDs provided as arguments are to be considered non-background,
            # even if a the ID is in the background set. All other IDs in the
            # union then must be from the background set.
            focus_id2isbg = []
            for idx in focus_indices:
                cid = self._row_id_index_map[idx]
                focus_id2isbg.append(False if cid in clip_ids else True)

            # Create new matrix from row-column intersections
            ret_mat = self._kernel[focus_indices, :][:, focus_indices]

            return focus_clipids, focus_id2isbg, ret_mat

    # noinspection PyPep8Naming
    def extract_rows(self, clipID_or_IDs, col_ids=None):
        """
        Find and return the v-stacked distance vectors, in kernel row order
        (i.e. not in the order given as arguments), of the kernel rows matching
        the given clip IDs.

        Note: The matrix returned will always be a new instance and not set up
        to use shared memory. When directly used with shared memory objects, it
        will be passed by value, not by reference.

        :param clipID_or_IDs: The integer clip ID or IDs of which to get the
            distance vectors for.
        :type clipID_or_IDs: int or Iterable of int

        :param col_ids: Select these IDs along the column axis. If None, select
            all IDs along the column axis.
        :type col_ids: None or Iterable of int

        :return: The row-wise index-to-clipID map (tuple), the column-wise
            index-to-clipID map (tuple), and the KxL shape matrix, where K is
            the number of clip IDs given to the method, and L is the width
            (columns) of the distance kernel.
        :rtype: tuple of int, tuple of int, matrix

        """
        with self._rw_lock.read_lock():
            with SimpleTimer("Checking inputs", self._log):
                try:
                    clipID_or_IDs = frozenset(int(e) for e in clipID_or_IDs)
                except:
                    raise ValueError("Not all clip IDs could be used as ints!")

                id_diff = clipID_or_IDs.difference(self._row_id_index_map)
                assert not id_diff, \
                    "Not all clip IDs provided are represented in this " \
                    "distance kernel matrix! (difference: %s)" \
                    % id_diff

                if col_ids:
                    try:
                        col_ids = frozenset(int(e) for e in col_ids)
                    except:
                        raise ValueError("Not all column IDs could be cast to "
                                         "int!")
                    id_diff = col_ids.difference(self._col_id_index_map)
                    assert not id_diff, \
                        "Not all column IDs provided are represented in this " \
                        "distance kernel's column axis! (difference: %s" \
                        % id_diff

                del id_diff

            # Reorder the given clip IDs so that they are in the same relative
            # order as the kernel matrix edge order
            # TODO: Optimize this, its taking a while
            #       --> Might have optimized using sets? but added another focus
            #           element, so might not be any better or worse than before
            with SimpleTimer("Creating focus index/cid sequence", self._log):
                focus_row_indices = []
                focus_row_clipids = []
                for idx, cid in enumerate(self._row_id_index_map):
                    if cid in clipID_or_IDs:
                        focus_row_indices.append(idx)
                        focus_row_clipids.append(cid)
                focus_col_indices = []
                focus_col_clipids = []
                if col_ids:
                    for idx, cid in enumerate(self._col_id_index_map):
                        if cid in col_ids:
                            focus_col_indices.append(idx)
                            focus_col_clipids.append(cid)
                else:
                    focus_col_indices = range(len(self._col_id_index_map))
                    focus_col_clipids = self._col_id_index_map

            with SimpleTimer("Cropping kernel to focus range", self._log):
                return (
                    tuple(focus_row_clipids), tuple(focus_col_clipids),
                    self._kernel[focus_row_indices, :][:, focus_col_indices]
                )


class FeatureMemory (object):
    """
    Class for encapsulating and managing feature and kernel matrices for
    different feature types
    """

    @classmethod
    def construct_from_files(cls, id_vector_file, bg_flags_file,
                             feature_mat_file, kernel_mat_file, rw_lock=None):
        """ Initialize FeatureMemory object from file sources.

        :param id_vector_file: File containing the numpy.save(...) output of
            clip ID values in the order in which they associate to the rows of
            the kernel matrix.
        :type id_vector_file: str
        :param bg_flags_file: File containing output of numpy.save(...) where
            each index maps a row index of the kernel to whether or not the
            associated clip ID should be considered a background video or not.
        :type bg_flags_file: str
        :param feature_mat_file: File containing the kernel matrix as saved by
            numpy.save(...) (saved as an ndarray, converted to matrix on load).
        :type feature_mat_file: str
        :param kernel_mat_file: File containing the kernel matrix as saved by
            numpy.save(...) (saved as an ndarray, converted to matrix on load).
        :type kernel_mat_file: str

        :return: Symmetric FeatureMemory constructed with the data provided in
            the provided files.
        :rtype: FeatureMemory

        """
        clip_ids = np.load(id_vector_file)
        bg_flags = np.load(bg_flags_file)
        # noinspection PyCallingNonCallable
        feature_mat = np.matrix(np.load(feature_mat_file))
        # noinspection PyCallingNonCallable
        kernel_mat = np.matrix(np.load(kernel_mat_file))

        bg_clips = set([clip_ids[i]
                        for i, f in enumerate(bg_flags)
                        if f])

        return FeatureMemory(clip_ids, bg_clips, feature_mat, kernel_mat,
                             rw_lock=rw_lock)

    @classmethod
    def construct_from_descriptor(cls, feature_descriptor, rw_lock=None):
        """
        Initialize FeatureMemory object from file sources defined by a
        FeatureDescriptor

        :param feature_descriptor:
        :type feature_descriptor: masir.search.FeatureDescriptor.FeatureDescriptor

        :return: Symmetric FeatureMemory constructed with the data provided in
            the provided files.
        :rtype: FeatureMemory

        """
        # noinspection PyTypeChecker
        return FeatureMemory.construct_from_files(
            feature_descriptor.ids_file, feature_descriptor.bg_flags_file,
            feature_descriptor.feature_data_file,
            feature_descriptor.kernel_data_file,
            rw_lock=rw_lock
        )

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __init__(self, id_vector, bg_clip_ids, feature_mat, kernel_mat,
                 rw_lock=None):
        """ Initialize this FeatureMemory object

        This class must be used with numpy ndarray and matrix classes for shared
        memory purposes.

        NOTE: Arrays and matrices given here must own their data! This is
        currently required in order to resize them later when updating with new
        feature vectors. A ValueError will be thrown if an given array/matrix
        does not own its data.

        TODO: Allow kernel matrix to be optional, causing it to be built from
        the provided feature matrix (not a recommended action).

        :param id_vector: (numpy) Array of clip IDs. This is used as the map
            from an index position to the clip ID its associated with in the
            kernel and distance kernel matrices.
        :type id_vector: ndarray of int
        :param bg_clip_ids: (numpy) Array of clip IDs that are to be treated as
            background clip IDs.
        :type bg_clip_ids: set of int
        :param feature_mat: (numpy) Matrix of features for clip IDs. Features
            should be stored vertically, i.e. Each row is a feature for a
            particular clip ID (id_vector being the index-to-clipID map).
        :type feature_mat: matrix of float
        :param kernel_mat: (numpy) Matrix detailing the distances between
            feature vectors. This must be a square, symmetric matrix.
        :type kernel_mat: matrix of float
        :param rw_lock: Optional ReadWriteLock for this instance to use. If not
            provided, we will create our own.
        :type rw_lock: None or ReadWriteLock

        """
        assert isinstance(id_vector, (ndarray, ArrayProxy)), \
            "ID vector not given as a numpy.ndarray!"
        assert isinstance(bg_clip_ids, (set, frozenset)), \
            "Background ID vector not a set!"
        assert isinstance(feature_mat, (matrix, MatrixProxy)), \
            "Kernel matrix not a numpy.matrix!"
        assert isinstance(kernel_mat, (matrix, MatrixProxy)), \
            "Distance kernel not a numpy.matrix!"

        # noinspection PyUnresolvedReferences
        # -> base IS a member of the matrix class...
        if id_vector.base is not None:
            raise ValueError("Given ``id_vector`` does not own its data! It "
                             "will not be transformable later.")
        elif feature_mat.base is not None:
            raise ValueError("Given ``feature_mat`` does not own its data! It "
                             "will not be transformable later.")
        elif kernel_mat.base is not None:
            raise ValueError("Given ``kernel_mat`` does not own its data! It "
                             "will not be transformable later.")

        # The kernel should be square and should be the same size as the feature
        # matrix's number or rows (unique stored clip features).
        if not (kernel_mat.shape[0] == kernel_mat.shape[1] == feature_mat.shape[0]):
            raise ValueError("The distance kernel matrix provided is either "
                             "misshapen or conflicts with the dimensions of "
                             "the provided feature matrix. (kernel matrix "
                             "shape: %s, num feature vectors: %d"
                             % (kernel_mat.shape, feature_mat.shape[0]))

        #self._log.debug("Lock given: %s", rw_lock)
        if rw_lock:
            assert isinstance(rw_lock, ReadWriteLock), \
                "Not given a value ReadWriteLock instance!"
            self._rw_lock = rw_lock
        else:
            #self._log.debug("Falling back on bad lock given (given: %s)",
            #                type(rw_lock))
            self._rw_lock = ReadWriteLock()

        self._id_vector = id_vector
        self._bg_clip_ids = bg_clip_ids
        self._feature_mat = feature_mat
        self._kernel_mat = kernel_mat

        # Helper structure mapping clipIDs to their row index
        self._cid2idx_map = dict((cid, idx) for idx, cid
                                 in enumerate(self._id_vector))

    @staticmethod
    def _histogram_intersection_distance(a, b):
        """
        Calculates distance between two vectors using histogram intersection.

        Non-branching version of the histogram intersection algorithm.

        :param a: A vector in array form.
        :type a: ndarray
        :param b: A vector in array form.
        :type b: ndarray

        :return: Histogram Intersection (HI) distance scalar
        :rtype: double

        """
        # branching version
        #return np.vstack((a, b)).min(axis=0).sum()

        # Non-branching version
        # noinspection PyUnresolvedReferences
        return (a + b - np.abs(a - b)).sum() * 0.5

    def _generate_distance_kernel_matrix(self):
        """
        Helper method to generate a distance kernel from the kernel matrix (HIK)

        This takes a long time...

        """
        with self._rw_lock.read_lock():
            # Create matrix whose elements are the distances between all row
            # permutations
            fmat = self._feature_mat  # shorter name
            num_rows = fmat.shape[0]

            # distance kernel is a square matrix based on feature samples
            dist_kernel = np.mat(np.ndarray((num_rows,)*2))
            self._log.info("Creating distance kernel with shape %s",
                           dist_kernel.shape)

            timer_log = logging.getLogger('.'.join((self.__module__,
                                                    self.__class__.__name__,
                                                    "SimpleTimer")))

            for i in xrange(num_rows - 1):
                with SimpleTimer('computing distances from row %d to [%d-%d]'
                                 % (i, i+1, num_rows-1), timer_log):
                    dist_kernel[i, i] = 1.0
                    for j in xrange(i + 1, num_rows):
                        dist = self._histogram_intersection_distance(fmat[i],
                                                                     fmat[j])
                        dist_kernel[i, j] = dist_kernel[j, i] = dist
            dist_kernel[-1, -1] = 1.0
            return dist_kernel

    def get_ids(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        with this object's lock when in a parallel environment to prevent
        possible memory corruption.

        :return: Ordered vector of clip IDs along the row-edge of this object's
            feature matrix and along both edges of the kernel matrix.
        :rtype: ndarray

        """
        return self._id_vector

    def get_bg_ids(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        with this object's lock when in a parallel environment to prevent
        possible memory corruption.

        :return: Ordered vector of clip IDs that we are treating as background
            clips.
        :rtype: frozenset of int

        """
        return frozenset(self._bg_clip_ids)

    def get_feature_matrix(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        possible memory corruption.

        :return: Matrix recording feature vectors for a feature type. See the
            id vector for row-wise index-to-clipID association.
        :rtype: matrix

        """
        return self._feature_mat

    def get_kernel_matrix(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        with this object's lock when in a parallel environment to prevent
        possible memory corruption.

        :return: Symmetric matrix detailing the distances between any two clip
            ID features. Distances are computed via histogram intersection.
        :rtype: matrix

        """
        return self._kernel_mat

    def get_lock(self):
        """
        :return: a reference to this object's read/write lock.
        :rtype: ReadWriteLock

        """
        return self._rw_lock

    def get_distance_kernel(self):
        """
        DistanceKernel object constructed from this feature's current state.

        :return: This feature distance kernel.
        :rtype: DistanceKernel

        """
        with self._rw_lock.read_lock():
            return DistanceKernel(self._id_vector, self._id_vector,
                                  self._kernel_mat, self._bg_clip_ids,
                                  self._rw_lock)

    def get_feature(self, *clip_id_or_ids):
        """
        Return the a matrix where each row is the feature vector for one or more
        clip IDs. The given list of clip IDs given acts as the index-to-clipID
        map for the returned matrix's rows. If repeat clip IDs are provided in
        the input, there will be repeat feature vectors in the returned matrix.

        Raises ValueError if the given clip ID is not represented in the current
        matrix.

        :param clip_id_or_ids: One or more integer clip IDs to retrieve the
            feature vectors for.
        :type clip_id_or_ids: tuple of int

        :return: NxM matrix, where N is the number of clip IDs requested and M
            is the length of a feature vector for this vector.
        :rtype: np.matrix

        """
        assert all(isinstance(e, int) for e in clip_id_or_ids), \
            "Not given an integer or a valid iterable over integers!"

        with self._rw_lock.read_lock():
            # rows = num of IDs given, cols = width of feature matrix
            with SimpleTimer("Allocating return matrix", self._log):
                # noinspection PyUnresolvedReferences
                # -> matrix class DOES have ``dtype`` property...
                ret_mat = matrix(ndarray((len(clip_id_or_ids),
                                          self._feature_mat.shape[1]),
                                         self._feature_mat.dtype))
            for i, cid in enumerate(clip_id_or_ids):
                feature_idx = self._cid2idx_map[cid]
                ret_mat[i, :] = self._feature_mat[feature_idx, :]
            return ret_mat

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def update(self, clip_id, feature_vec=None, is_background=False,
               timeout=None):
        """
        Update this feature with a feature vector associated with a clip ID. If
        clip ID is already in the feature matrix, we replace the current vector
        with the given one.

        Either way, the distance kernel is updated with either a new row/column,
        or updating relevant slots in the existing distance kernel.

        :raise ValueError: if the given feature vector is not compatible with
            our feature vector.
        :raise RuntimeError: If a timeout is given and the underlying write lock
            doesn't acquire in that amount of time.

        :param clip_id: The ID of the clip the given ``feature_vec`` represents.
        :type clip_id: int
        :param feature_vec: Feature vector associated to the given clip ID. May
            be None if we are updating the background status of an existing clip
            ID.
        :type feature_vec: ndarray
        :param is_background: Flag declaring that this clip ID represents a
            background feature.
        :type is_background: bool
        :param timeout: Timeout seconds for the underlying write lock to acquire
            before a RuntimeError is thrown.
        :type timeout: None or int or float

        """
        with self._rw_lock.write_lock(timeout):
            clip_id = int(clip_id)
            if feature_vec is not None and \
                    not (feature_vec.ndim == 1
                         and len(feature_vec) == self._feature_mat.shape[1]):
                raise ValueError("Given feature vector not compatible "
                                 "(dimensionality or length does not match)")

            # Update the given feature vector and kernel distances
            # if self._cid2idx_map.get(clip_id, None) is not None:
            if clip_id in self._cid2idx_map:
                # In all cases, update the background status of the clip
                if is_background:
                    self._bg_clip_ids.add(clip_id)
                else:
                    self._bg_clip_ids.discard(clip_id)

                # If we were given a new feature vector, update entries
                if feature_vec is not None:
                    idx = self._cid2idx_map[clip_id]
                    self._feature_mat[idx] = feature_vec
                    new_dist = np.mat(tuple(
                        self._histogram_intersection_distance(feature_vec, fv)
                        for fv in self._feature_mat
                    ))
                    self._kernel_mat[idx, :] = new_dist
                    self._kernel_mat[:, idx] = new_dist

            # Given a new clip id to add, must have feature to add.
            else:
                if feature_vec is None:
                    raise ValueError("Update given a new clip ID, but no "
                                     "feature vector. Feature vectors are "
                                     "required with new IDs.")

                # Update internal feature matrix with added vector
                self._cid2idx_map[clip_id] = self._id_vector.size
                self._id_vector.resize((self._id_vector.size + 1,),
                                       refcheck=False)
                self._id_vector[-1] = clip_id

                if is_background:
                    self._bg_clip_ids.add(clip_id)

                # noinspection PyUnresolvedReferences
                if self._feature_mat.base is not None:
                    raise RuntimeError("Feature matrix does not own its data")
                # Since we're only adding a new row, this resize does not affect
                # the positioning of the existing data.
                self._log.debug("Updating feature matrix...")
                # noinspection PyUnresolvedReferences
                self._feature_mat.resize((self._feature_mat.shape[0] + 1,
                                          self._feature_mat.shape[1]),
                                         refcheck=False
                                         )
                self._feature_mat[-1, :] = feature_vec

                # Need to add a new row AND column to the distance kernel.
                self._log.debug("Updating kernel matrix...")
                if self._kernel_mat.base is not None:
                    raise RuntimeError("kernel matrix does not own its data")
                assert self._kernel_mat.shape[0] == self._kernel_mat.shape[1], \
                    "kernel matrix is not symmetric for some reason???"
                # noinspection PyPep8Naming
                # -> because I like ``N`` better...
                N = self._kernel_mat.shape[0]
                kernel_copy = np.matrix(self._kernel_mat)
                self._kernel_mat.resize((N+1, N+1), refcheck=False)
                self._kernel_mat[:N, :N] = kernel_copy
                del kernel_copy

                # Computing new feature distance (histogram intersection). Only
                # need to compute this once because of HI being being
                # commutative and the kernel matrix being symmetric.
                self._log.debug("Adding new distance vectors...")
                dist_vec = np.mat(tuple(
                    self._histogram_intersection_distance(feature_vec, fv)
                    for fv in self._feature_mat
                ))
                self._kernel_mat[-1, :] = dist_vec
                self._kernel_mat[:, -1] = dist_vec.T


class FeatureMemoryMap (object):
    """ Map different feature types to their own FeatureMemory object
    """
    # Basically a pass-through for all the functions in FeatureMemory, but with
    # and additional heading parameter of feature type.

    def __init__(self):
        # self._map_lock = multiprocessing.RLock()
        self._map_lock = ReadWriteLock()
        #: :type: dict of (str, FeatureMemory)
        self._feature2memory = {}

    def get_feature_types(self):
        """ Get available feature types in this map.

        :return: Tuple of string names of all features initialize in this map
        :rtype: tuple of str

        """
        with self._map_lock.read_lock():
            return self._feature2memory.keys()

    def initialize(self, feature_type, id_vector, bg_clip_ids_vector,
                   feature_mat, kernel_mat, rw_lock=None):
        """ Initialize a feature type within this map

        :raise KeyError: When the given feature_type is already present in the
            map (requires a removal first).
        :raise ValueError: When there is an issue with one or more of the
            provided input data elements.

        :param feature_type: The name assigned to the feature type
        :param id_vector: Numpy array mapping indices to a clip ID.
        :param bg_clip_ids_vector: Numpy array of integers detailing the clip
            IDs that are to be considered background.
        :param feature_mat: Numpy matrix of all clip feature vectors. This
            should be an order 'C' matrix with features stacked vertically. The
            ``id_vector`` maps row indices to the clip ID the feature
            represents.
        :param kernel_mat: Pre-computed distance kernel

        """
        with self._map_lock.write_lock():
            # KeyError on repeat feature_type key, require removal first
            if feature_type in self._feature2memory:
                raise KeyError("Key '%s' already present in our mapping. "
                               "Please remove first before initializing."
                               % feature_type)

            self._feature2memory[feature_type] = \
                FeatureMemory(id_vector, bg_clip_ids_vector, feature_mat,
                              kernel_mat, rw_lock)

    def initialize_from_files(self, feature_type, id_vector_file, bg_flags_file,
                              feature_mat_file, kernel_mat_file, rw_lock=None):
        """ Initialize a feature type within this map from file resources.

        Files pointed to must be of the following formats:

            - id_vector_file:
                File resulting from a numpy.savetxt() of a one-dimensional
                array, mapping index position with an integer clip ID. This
                should correlate to the clip IDs of the row-major features
                stored in feature_mat_file.

            - bg_flags_file:
                File resulting from a numpy.savetxt() of a one-dimensional
                array, mapping index position with whether that clip should be
                treated as a background video or not. This should corrolate with
                features in the same way as the id_id_vector_file.

            - feature_mat_file:
                File resulting from a numpy.save() of an ndarray. This will be
                loaded in as a matrix. This should be the initial NxD feature
                matrix for this feature type.

            - kernel_mat_file:
                File resulting from a numpy.save() of an ndarray. This will be
                loaded in as a matrix. This should be the computed NxN distance
                kernel for this feature type.

        :param feature_type: The name assigned to the feature type.
        :type id_vector_file: str
        :type bg_flags_file: str
        :type feature_mat_file: str
        :type kernel_mat_file: str
        :param rw_lock: Optionally specified ReadWriteLock instance to use with
            the underlying FeatureMemory and DistanceKernel objects associated
            with this feature type (not recommended)
        :type rw_lock: ReadWriteLock

        """
        with self._map_lock.write_lock():
            # even though this happens in the initialize() call we make here,
            # we would like to short circuit before loading data if we can.
            if feature_type in self._feature2memory:
                raise KeyError("Key '%s' already present in our mapping. "
                               "Please remove first before initializing.")

            self._feature2memory[feature_type] = \
                FeatureMemory.construct_from_files(id_vector_file,
                                                   bg_flags_file,
                                                   feature_mat_file,
                                                   kernel_mat_file,
                                                   rw_lock)

    def remove(self, feature_type):
        """ Removes a feature type from our map, releasing its contents.

        :raise KeyError: If the given feature type does not currently map to
            anything.

        :param feature_type: The feature type to get the memory object of.
        :type feature_type: str

        """
        with self._map_lock.write_lock():
            del self._feature2memory[feature_type]

    def get_feature_memory(self, feature_type):
        """ Get the underlying FeatureMemory object for a feature type.

        :raise KeyError: If the given feature type does not currently map to
            anything.

        :param feature_type: The feature type to get the memory object of.
        :type feature_type: str

        :return: FeatureMemory object associated to the given feature type.
        :rtype: FeatureMemory

        """
        with self._map_lock.read_lock():
            return self._feature2memory[feature_type]

    def get_distance_kernel(self, feature_type):
        """ Get the DistanceKernel for a feature type.

        :raise KeyError: If the given feature type does not currently map to
            anything.

        :param feature_type: The feature type to get the memory object of.
        :type feature_type: str

        """
        with self._map_lock.read_lock():
            return self._feature2memory[feature_type]\
                       .get_distance_kernel()

    def get_feature(self, feature_type, *clip_id_or_ids):
        """
        With respect to the given feature types, return a matrix where each row
        is the feature vector for one or more clip IDs. The given clip IDs acts
        as the index-to-clipID map for the returned matrix's rows. If repeat
        clip IDs are provided in the input, there will be repeat feature vectors
        in the returned matrix.

        :raise ValueError: If the given clip ID is not represented in the
            feature matrix.

        :param clip_id_or_ids: One or more integer clip IDs to retrieve the
            feature vectors for.
        :type clip_id_or_ids: tuple of int

        :return: NxM matrix, where N is the number of clip IDs requested and M
            is the length of a feature vector for this vector.
        :rtype: np.matrix

        """
        with self._map_lock.read_lock():
            return self._feature2memory[feature_type]\
                       .get_feature(*clip_id_or_ids)

    def update(self, feature_type, clip_id, feature_vector, is_background=False,
               timeout=None):
        with self._map_lock.read_lock():
            return self._feature2memory[feature_type]\
                       .update(clip_id, feature_vector, is_background, timeout)


class TimedCache (object):
    """
    Be able to generally store FeatureMemory/DistanceKernel objects
    """

    def __init__(self):
        self._lock = multiprocessing.RLock()

        # Mapping of a key to an object
        #: :type: dict of (str, object)
        self._obj_cache = {}

        # Mapping of a key to the last access time for that object (UNIX time)
        #: :type: dict of (str, float)
        self._obj_last_access = {}

        # Mapping of timeout values for each entry
        self._obj_timeouts = {}

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def _check_expire(self):
        """
        Check all stored objects to see what has expired. Expired items are
        removed from our mappings.

        ASSUMING write lock has already been internally acquired.

        """
        self._log.debug("Checking entry expiration...")
        current_time = time.time()
        for key in self._obj_cache.keys():
            self._log.debug('  -> %s (type = %s)',
                            key, type(self._obj_cache[key]))
            # Remove if the key has a timeout, and the timeout period has been
            # exceeded (last access + timeout period <= current_time).
            if self._obj_timeouts[key] > 0 \
                    and current_time >= (self._obj_last_access[key]
                                         + self._obj_timeouts[key]):
                self._log.debug('     EXPIRED -- removing')
                # delete
                del self._obj_cache[key]
                del self._obj_last_access[key]
                del self._obj_timeouts[key]

    def get_rlock(self):
        return self._lock

    def keys(self):
        with self._lock:
            self._check_expire()
            return self._obj_cache.keys()

    def store(self, key, obj, timeout=0):
        """
        Store an object with a key. A timeout may be provided (in seconds).
        Defaults to no expiry.
        """
        with self._lock:
            self._check_expire()

            self._log.debug("storing '%s' (type=%s) with TO=%f",
                            key, type(obj), timeout)
            self._obj_cache[key] = obj
            self._obj_last_access[key] = time.time()
            self._obj_timeouts[key] = timeout

    def store_FeatureMemory(self, key, id_vector_file, bg_flags_file,
                            feature_mat_file, kernel_mat_file, timeout=0):
        fm = FeatureMemory.construct_from_files(id_vector_file,
                                                bg_flags_file,
                                                feature_mat_file,
                                                kernel_mat_file)
        self.store(key, fm, timeout)

    def store_DistanceKernel_symmetric(self, key, id_vector_file,
                                       kernel_mat_file, bg_flags_file=None,
                                       timeout=0):
        dk = DistanceKernel.construct_symmetric_from_files(id_vector_file,
                                                           kernel_mat_file,
                                                           bg_flags_file)
        self.store(key, dk, timeout)

    def store_DistanceKernel_asymmeric(self, key, row_ids_file, col_ids_file,
                                       kernel_mat_file, timeout=0):
        dk = DistanceKernel.construct_asymmetric_from_files(row_ids_file,
                                                            col_ids_file,
                                                            kernel_mat_file)
        self.store(key, dk, timeout)

    def get(self, key):
        """
        General getter function. When used with FeatureManager, I think this
        returns a copy of the remotely stored data.
        """
        with self._lock:
            self._check_expire()

            obj = self._obj_cache[key]
            self._log.debug("getting object '%s' (type=%s). "
                            "updating access time.",
                            key, type(obj))
            self._obj_last_access[key] = time.time()
            return obj

    def get_fm(self, key):
        obj = self.get(key)
        if isinstance(obj, FeatureMemory):
            return obj
        else:
            raise ValueError("Key not associated to a FeatureMemory object")

    def get_dk(self, key):
        obj = self.get(key)
        if isinstance(obj, DistanceKernel):
            return obj
        else:
            raise ValueError("Key not associated to a DistanceKernel object")

    def remove(self, key):
        """
        Remove a cached element if it exists. If it doesn't this is a no-op.
        """
        with self._lock:
            self._check_expire()

            if key in self._obj_cache:
                self._log.debug("removing entry '%s' (type=%s)",
                                key, type(self._obj_cache[key]))
                del self._obj_cache[key]
                del self._obj_last_access[key]
                del self._obj_timeouts[key]


# Create singleton objects if they don't exist a the time of import
if '__singleton_fmm__' not in globals():
    __singleton_fmm__ = FeatureMemoryMap()

if '__singleton_tc__' not in globals():
    __singleton_tc__ = TimedCache()


def get_common_fmm():
    """ Get global singleton FeatureMemoryMap

    :return: Singleton instance of the common FeatureMemoryMap
    :rtype: FeatureMemoryMap

    """
    return __singleton_fmm__


def get_common_tc():
    """ Get global singleton TimedCache

    :return: Singleton instance of common TimedCache
    :rtype: TimedCache

    """
    return __singleton_tc__


#
# Proxy Types and Support
#

class BaseProxy2 (BaseProxy):
    """
    Intermediate class under BaseProxy in order to fix the resetting of the
    _manager on fork (doesn't need to be?).
    """

    def __init__(self, token, serializer, manager=None, authkey=None,
                 exposed=None, incref=True):
        super(BaseProxy2, self).__init__(token, serializer, manager, authkey,
                                         exposed, incref)
        # Keep a second manager reference around to survive the base class's
        # after-fork nuke of the _manager attribute.
        self._manager_copy = manager

        multiprocessing.util.register_after_fork(self, BaseProxy2._after_fork2)

    def _after_fork2(self):
        self._manager = self._manager_copy

    def __reduce__(self):
        ret = super(BaseProxy2, self).__reduce__()
        ret[1][-1]['authkey'] = str(self._authkey)
        return ret


def all_properties(obj):
    """
    Return a list of names of non-methods of 'obj'
    """
    noncallables = []
    for name in dir(obj):
        if not hasattr(getattr(obj, name), '__call__'):
            noncallables.append(name)
    return noncallables


# Attributes that should not be overridden in generated proxy objects
__leave_alone_methods = frozenset((
    '__class__', '__delattr__', '__getattribute__', '__init__', '__metaclass__',
    '__new__', '__reduce__', '__reduce_ex__', '__setattr__', '__sizeof__',
    '__subclasshook__',
))


def all_safe_methods(obj):
    """
    Return the tuple of all "safe" method names of the given class. A "safe"
    method name is one that does not belong to the set
    ``__leave_alone_methods``. It is known that overwriting these callable
    methods interferes with the functioning of Proxy objects.

    :param obj: Class to extract callable methods from.
    :type obj: object

    :return: Tuple of safe method names to expose.
    :rtype: tuple of str

    """
    return tuple(set(all_methods(obj)).difference(__leave_alone_methods))


__leave_alone_properties = frozenset((
    '__dict__', '__module__'
))


def all_safe_properties(obj):
    """
    Return the tuple of all "safe" non-callable elements of the given class. A
    "safe" non-callable is one that does not belong to the set
    ``__leave_alone_properties``. It is known that overwriting these
    non-callables interferes with the functioning of Proxy objects.

    :param obj: Class to extract non-callable properties from.
    :type obj: object

    :return: Tuple of safe non-callable property names to expose.
    :rtype: tuple of str

    """
    return tuple(set(all_properties(obj)).difference(__leave_alone_properties))


class ExposedAutoGenMeta (type):

    def __new__(mcs, clsname, bases, dct):
        # look at the _exposed_
        to_expose = set(dct.get('_exposed_', None))
        exposed = set()
        if to_expose is None:
            raise ValueError("Classes using metaclass 'ExposedAutoGenMeta' "
                             "MUST have an ``_exposed_`` iterable defined.")

        # If any of the methods in to_expose are defined in dct, remove them
        # from to_expose and add them to the already exposed set.
        for name in set.intersection(to_expose, dct.keys()):
            if not hasattr(dct[name], '__call__'):
                raise ValueError("Declared an _exposed_ method '%s', but "
                                 "manually defined the same symbol as non-"
                                 "callable." % name)
            to_expose.remove(name)
            exposed.add(name)

        # If the class has a _exposed_properties_ iterable, add __*attr__
        # pass-throughs
        if dct.get('_exposed_properties_', None):
            # remove duplicates if any present
            dct['_exposed_properties_'] = \
                tuple(set(dct['_exposed_properties_']))
            exposed.update(('__getattribute__', '__setattr__', '__delattr__'))

            if '__getattribute__' in to_expose:
                to_expose.remove('__getattribute__')
            if '__getattribute__' in dct:
                print ("WARNING: ExposedAutoGenMeta overwriting custom "
                       "``__getattribute__`` in class '%s' in favor of "
                       "property proxy supporting version."
                       % clsname)
            exec """def __getattr__(self, key):
                if key in self._exposed_properties_:
                    callmethod = object.__getattribute__(self, '_callmethod')
                    return callmethod('__getattribute__', (key,))
                return object.__getattribute__(self, key)""" in dct

            if '__setattr__' in to_expose:
                to_expose.remove('__setattr__')
            if '__setattr__' in dct:
                print ("WARNING: ExposedAutoGenMeta overwriting custom "
                       "``__setattr__`` in class '%s' in favor of "
                       "property proxy supporting version."
                       % clsname)
            exec """def __setattr__(self, key, value):
                if key in self._exposed_properties_:
                    callmethod = object.__getattribute__(self, '_callmethod')
                    return callmethod('__setattr__', (key,))
                return object.__setattr__(self, key, value)""" in dct

            if '__delattr__' in to_expose:
                to_expose.remove('__delattr__')
            if '__delattr__' in dct:
                print ("WARNING: ExposedAutoGenMeta overwriting custom "
                       "``__delattr__`` in class '%s' in favor of "
                       "property proxy supporting version."
                       % clsname)
            exec """def __delattr__(self, key):
                if key in self._exposed_properties_:
                    callmethod = object.__getattribute__(self, '_callmethod')
                    return callmethod('__delattr__', (key,))
                return object.__delattr__(self, key)""" in dct

        # Create default method stamps for remaining methods.
        for method in to_expose:
            exec '''def %s(self, *args, **kwds):
            return self._callmethod(%r, args, kwds)''' % (method, method) in dct
            exposed.add(method)

        # Refresh class var with those methods that have been exposed
        dct['_exposed_'] = tuple(sorted(exposed))

        return super(ExposedAutoGenMeta, mcs).__new__(mcs, clsname, bases, dct)


# Wrapper around numpy ndarray class, providing pass-through functions for class
# functions as of numpy 1.8.0
# TODO: Add``method_to_typeid`` for functions that return copies of data
class ArrayProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(ndarray)
    _exposed_properties_ = all_safe_properties(ndarray)
    _method_to_typeid_ = {
        '__iter__': "Iterator"
    }


# Wrapper around numpy matrix class, providing pass-through functions for class
# functions as of numpy 1.8.0.
# TODO: Add``method_to_typeid`` for functions that return copies of data
class MatrixProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(np.matrix)
    _exposed_properties_ = all_safe_properties(matrix)
    _method_to_typeid_ = {
        '__iter__': "Iterator"
    }


class RWLockWithProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = ('__enter__', '__exit__')


# noinspection PyPep8Naming
class ReadWriteLockProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(ReadWriteLock)
    _method_to_typeid_ = {
        'read_lock':    'rRWLockWith',
        'write_lock':   'rRWLockWith'
    }

    def acquireRead(self, timeout=None, _id=None):
        # Overwriting userID lock uses as the caller of this proxy method.
        me = _id or (current_process().ident, current_thread().ident)
        return self._callmethod('acquireRead', (timeout, me))

    def acquireWrite(self, timeout=None, _id=None):
        # Overwriting userID lock uses as the caller of this proxy method.
        me = _id or (current_process().ident, current_thread().ident)
        return self._callmethod('acquireWrite', (timeout, me))

    def releaseRead(self, _id=None):
        # Overwriting userID lock uses as the caller of this proxy method.
        me = _id or (current_process().ident, current_thread().ident)
        return self._callmethod('releaseRead', (me,))

    def releaseWrite(self, _id=None):
        # Overwriting userID lock uses as the caller of this proxy method.
        me = _id or (current_process().ident, current_thread().ident)
        return self._callmethod('releaseWrite', (me,))

    def read_lock(self, timeout=None, _id=None):
        me = _id or (current_process().ident, current_thread().ident)
        return self._callmethod('read_lock', (timeout, me))

    def write_lock(self, timeout=None, _id=None):
        me = _id or (current_process().ident, current_thread().ident)
        return self._callmethod('write_lock', (timeout, me))


class DistanceKernelProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(DistanceKernel)
    _method_to_typeid_ = {
        'get_lock':             'rReadWriteLock',
        'get_kernel_matrix':    'rmatrix',
    }


class FeatureMemoryProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(FeatureMemory)
    _method_to_typeid_ = {
        '_generate_distance_kernel_matrix': 'rmatrix',

        'get_feature_matrix':   'rmatrix',
        'get_kernel_matrix':    'rmatrix',
        'get_lock':             'rReadWriteLock',

        'get_distance_kernel':  'rDistanceKernel',
        'get_feature':          'rmatrix',
    }


class FeatureMemoryMapProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(FeatureMemoryMap)
    _method_to_typeid_ = {
        'get_feature_memory':   "rFeatureMemory",
        'get_distance_kernel':  "rDistanceKernel",
        'get_feature':          "rmatrix",
    }


class TimedCacheProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(TimedCache)
    _method_to_typeid_ = {
        'get_rlock':    'RLock',
        'get_fm':       'rFeatureMemory',
        'get_dk':       'rDistanceKernel',
    }


#
# FeatureManager and Type registration
#
class FeatureManager (SyncManager):
    """
    This class shouldn't be initialized directly, but instead initialized and
    retrieved through the init and get functions below.
    """
    pass


# Object-based
FeatureManager.register('array',            np.array,         ArrayProxy)
FeatureManager.register('ndarray',          np.ndarray,       ArrayProxy)
FeatureManager.register('matrix',           np.matrix,        MatrixProxy)
FeatureManager.register('ReadWriteLock',    ReadWriteLock,    ReadWriteLockProxy)
FeatureManager.register('DistanceKernel',   DistanceKernel,   DistanceKernelProxy)
FeatureManager.register('FeatureMemory',    FeatureMemory,    FeatureMemoryProxy)
FeatureManager.register('FeatureMemoryMap', FeatureMemoryMap, FeatureMemoryMapProxy)
# Function based
FeatureManager.register('get_common_fmm',
                        get_common_fmm,
                        FeatureMemoryMapProxy)
FeatureManager.register('get_common_tc',
                        get_common_tc,
                        TimedCacheProxy)
FeatureManager.register("symmetric_dk_from_file",
                        DistanceKernel.construct_symmetric_from_files,
                        DistanceKernelProxy)
FeatureManager.register("asymmetric_dk_from_file",
                        DistanceKernel.construct_asymmetric_from_files,
                        DistanceKernelProxy)

# Return proxy registrations
# - If a function Proxy has a method that should return another proxy, the
#   proxy type that it returns CANNOT have been registered with
#   ``create_method=True``. Reason: the server uses the registered typeid's
#   callable to create the return object
FeatureManager.register('rarray',           proxytype=ArrayProxy,           create_method=False)
FeatureManager.register("rmatrix",          proxytype=MatrixProxy,          create_method=False)
# Generated structure from read_lock() and write_lock() methods in ReadWriteLock
FeatureManager.register("rRWLockWith",      proxytype=RWLockWithProxy,      create_method=False)
FeatureManager.register("rReadWriteLock",   proxytype=ReadWriteLockProxy,   create_method=False)
FeatureManager.register("rDistanceKernel",  proxytype=DistanceKernelProxy,  create_method=False)
FeatureManager.register("rFeatureMemory",   proxytype=FeatureMemoryProxy,   create_method=False)


# Map of FeatureManager instance keyed on the address
#: :type: dict of (tuple or str or None, FeatureManager)
__mgr_cache__ = {}


# noinspection PyPep8Naming
def initFeatureManagerConnection(address=None, authkey=None,
                                 serializer='pickle'):
    """ Initialize FeatureManager connection

    :raises ValueError: The given address already maps to an initialized
        FeatureManager.

    :param address: The address of the FeatureManager server to connect to, or
        None for the local process tree's FeatureManager. When None, the server
        is a child process of this process, or the parent process that first
        created the local server.
    :type address: str or (str, int) or None

    :param authkey: The authentication key for the server we are connecting to.
        This is only used when address is None and this is the first time we
        are getting the feature manager for this process tree (creating the
        connection).
    :type authkey: None or str

    """
    global __mgr_cache__
    if address in __mgr_cache__:
        raise ValueError("The given address '%s' already maps to an "
                         "initialized FeatureManager!"
                         % str(address))
    __mgr_cache__[address] = FeatureManager(address, authkey, serializer)

    # if address is None, then we are initializing a local process tree server.
    # i.e. we need to fork and start it
    try:
        if address is None:
            __mgr_cache__[address].start()
        else:
            __mgr_cache__[address].connect()
    # If an error occurred, rollback what we just added
    except Exception:
        del __mgr_cache__[address]
        raise


def removeFeatureManagerConnection(address=None):
    """ Shutdown and remove an initialized FeatureManager connection
    :raises KeyError: if the given address is not associated with to an active
        manager.
    :param address: The address of the FeatureManager connection to remove from
        the active mapping.
    :type address: str or (str, int) or None
    """
    global __mgr_cache__
    #: :type: FeatureManager
    if hasattr(__mgr_cache__[address], 'shutdown'):
        __mgr_cache__[address].shutdown()
    del __mgr_cache__[address]


# noinspection PyPep8Naming
def getFeatureManager(address=None):
    """
    Get the FeatureManager instance for the given address from the cache. If the
    address is None, returns the FeatureManager for the current process tree
    (may have been initialized on a parent process).

    :raises KeyError: When the given address has not been initialized and is not
        present in the cache.

    :param address: The address of the FeatureManager connection to retrieve
        from the active mapping.
    :type address: str or (str, int) or None

    :return: Singleton feature_manager for the given address.
    :rtype: FeatureManager

    """
    return __mgr_cache__[address]
