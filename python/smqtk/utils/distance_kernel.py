"""
LICENCE
-------
Copyright 2013-2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

import logging
import numpy as np
from numpy.core.multiarray import ndarray  # for shortening doc strings

from smqtk.utils import ReadWriteLock
from smqtk.utils import SimpleTimer
from smqtk.utils.parallel import parallel_map
from smqtk.utils.bin_utils import report_progress


def compute_distance_kernel(m, dist_func, row_wise=False, parallel=True):
    """
    Method for computing the distance kernel of an array of vectors given a
    distance function that works on two supplied 1D arrays.

    For a valid distance function interface, see
    ``smqtk.utils.distance_functions.histogram_intersection_distance2``.

    :param m: An array of vectors to compute the pairwise distance kernel for.
    :type m: numpy.core.multiarray.ndarray

    :param dist_func: Distance function
    :type dist_func: (ndarray, ndarray) -> ndarray[float] | float

    :param row_wise: If the given distance function can take a vector and a
        matrix, and computes pair-wise distances, returning a vector of
        distances between the given vector and each row of the matrix.
    :type row_wise: bool

    :param parallel: If distances should be calculated in parallel. This is true
        by default.
    :type parallel: bool

    :return: Computed symmetric distance kernel
    :rtype: numpy.core.multiarray.ndarray

    """
    if hasattr(dist_func, 'im_func'):
        # noinspection PyUnresolvedReferences
        distance_name = '.'.join([dist_func.__module__,
                                  dist_func.im_class.__name__,
                                  dist_func.im_func.func_name])
    elif hasattr(dist_func, 'func_name'):
        # noinspection PyUnresolvedReferences
        distance_name = '.'.join([dist_func.__module__,
                                  dist_func.func_name])
    elif hasattr(dist_func, 'py_func') \
            and hasattr(dist_func.py_func, 'func_name'):
        distance_name = '.'.join([dist_func.__module__,
                                  dist_func.py_func.func_name])
    else:
        distance_name = "<unknown>"
    log = logging.getLogger(__name__)

    if m.ndim == 1:
        m = m[np.newaxis]

    log.info("Computing distance kernel")
    side = m.shape[0]
    mat = np.ndarray((side, side), dtype=float)

    s = [0] * 7
    if row_wise:
        log.debug("Computing row-wise distances")
        # For all rows except the last one. We'll have computed all distanced by
        # the time reach m[side-1]
        if parallel:
            def work_func(i):
                mat[i, i] = dist_func(m[i], m[i])
                if i < (side - 1):
                    mat[i + 1:, i] = mat[i, i + 1:] = dist_func(m[i, :],
                                                                m[i + 1:, :])
            # Using threading for in-place modification
            s = [0] * 7
            for _ in parallel_map(work_func, xrange(side),
                                  use_multiprocessing=False):
                report_progress(log.debug, s, 1.)
        else:
            for i in xrange(side):
                # Compute col/row wise distances
                mat[i, i] = dist_func(m[i], m[i])
                if i < (side-1):
                    mat[i+1:, i] = mat[i, i+1:] = dist_func(m[i, :], m[i+1:, :])
                report_progress(log.debug, s, 1.)
    else:
        log.debug("Computing element-wise distances")
        if parallel:
            def work_func(i):
                mat[i, i] = dist_func(m[i], m[i])
                # cols to the left of diagonal index for this row
                for j in xrange(i):
                    mat[i, j] = mat[j, i] = dist_func(m[i], m[j])
            # Using threading for in-place modification
            for _ in parallel_map(work_func, xrange(side),
                                  use_multiprocessing=False):
                report_progress(log.debug, s, 1.)
        else:
            for i in xrange(side):
                mat[i, i] = dist_func(m[i], m[i])
                # cols to the left of diagonal index for this row
                for j in xrange(i):
                    mat[i, j] = mat[j, i] = dist_func(m[i], m[j])
                report_progress(log.debug, s, 1.)

    return mat


def compute_distance_matrix(m1, m2, dist_func, row_wise=False):
    """
    Function for computing the pair-wise distance matrix between two arrays of
    vectors. Both matrices must have the same number of columns.
    """
    if m1.ndim == 1:
        m1 = m1[np.newaxis]
    if m2.ndim == 1:
        m2 = m2[np.newaxis]
    k = np.ndarray((m1.shape[0], m2.shape[0]), dtype=float)
    if row_wise:
        # row wise
        for i in xrange(m1.shape[0]):
            k[i, :] = dist_func(m1[i], m2)
    else:
        for i in xrange(m1.shape[0]):
            for j in xrange(m2.shape[0]):
                k[i, j] = dist_func(m1[i], m2[j])
    return k


class DistanceKernel (object):
    """
    Feature Distance Kernel object.

    This class allows the kernel to either be symmetric or not. If it is
    symmetric, the ``symmetric_submatrix`` function becomes available.

    Intended to be used with ProxyManager proxy objects (given at
    construction)

    MONKEY PATCHING:
    When using this object directly (not using the ProxyManager stuff) and
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
            bg_clips = np.array([clip_ids[i]
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
        :param bg_clip_ids: Optional array of boolean flags, marking whether an
            index should be considered a "background" video. Contents will be
            treated as ints.
        :type bg_clip_ids: set of int
        :param rw_lock: Read-Write lock for data provided. This should be
            provided if the any of the data is shared with other objects/
            sources. If this is given None (default), then a lock is created.
        :type rw_lock: ReadWriteLock or None

        """
        # TODO: Possibly add checks for the id arrays like there is for the
        #       bgclipid array (int-able contents)
        assert row_id_index_map.shape[0] == kernel_mat.shape[0], \
            "Length of row index map and kernel row count did not match! " \
            "(row index map: %d, kernel row count: %d)" \
            % (row_id_index_map.shape[0], kernel_mat.shape[0])
        assert col_id_index_map.shape[0] == kernel_mat.shape[1], \
            "Length of col index map and kernel col count did not match! " \
            "(col index map: %d, kernel col count: %d)" \
            % (col_id_index_map.shape[0], kernel_mat.shape[1])

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
        returned index-to-is-background mapping (tuple).

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
        :rtype: tuple of int, tuple of bool, numpy.matrixlib.defmatrix.matrix

        """
        with self._rw_lock.read_lock():
            with SimpleTimer("Checking inputs", self._log.debug):
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
                    raise ValueError("Not all clip IDs could be used as ints!")

                id_diff = set(clip_ids).difference(self._row_id_index_map)
                assert not id_diff, \
                    "Not all clip IDs provided are represented in this " \
                    "distance kernel matrix! (difference: %s)" \
                    % id_diff
                del id_diff

            with SimpleTimer("Computing union of BG clips and provided IDs",
                             self._log.debug):
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

            ret_mat = self._kernel[focus_indices, :][:, focus_indices]
            return focus_clipids, focus_id2isbg, ret_mat

    # noinspection PyPep8Naming
    def extract_rows(self, *clipID_or_IDs):
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

        :return: The row-wise index-to-clipID map (tuple), the column-wise
            index-to-clipID map (tuple), and the KxL shape matrix, where K is
            the number of clip IDs given to the method, and L is the width
            (columns) of the distance kernel.
        :rtype: tuple of int, tuple of int, matrix

        """
        with self._rw_lock.read_lock():
            with SimpleTimer("Checking inputs", self._log.debug):
                try:
                    clipID_or_IDs = frozenset(int(e) for e in clipID_or_IDs)
                except Exception, ex:
                    raise ValueError("Not all clip IDs could be used as ints: "
                                     "%s" % str(ex))

                id_diff = clipID_or_IDs.difference(self._row_id_index_map)
                assert not id_diff, \
                    "Not all clip IDs provided are represented in this " \
                    "distance kernel matrix! (difference: %s)" \
                    % id_diff
                del id_diff

            # Reorder the given clip IDs so that they are in the same relative
            # order as the kernel matrix edge order
            with SimpleTimer("Creating focus index/cid sequence", self._log.debug):
                focus_row_indices = []
                focus_row_clipids = []
                for idx, cid in enumerate(self._row_id_index_map):
                    # if ((cid in clipID_or_IDs)
                    #         and (cid not in focus_row_clipids)):
                    if cid in clipID_or_IDs:
                        focus_row_indices.append(idx)
                        focus_row_clipids.append(cid)

            with SimpleTimer("Cropping kernel to focus range", self._log.debug):
                return (
                    tuple(focus_row_clipids),
                    tuple(self._col_id_index_map),
                    self._kernel[focus_row_indices, :]
                )
