"""
LICENCE
-------
Copyright 2013-2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

import logging
import multiprocessing
import numpy as np

from smqtk.utils import ReadWriteLock
from smqtk.utils import SimpleTimer
from smqtk.utils.distance_kernel import DistanceKernel


class FeatureMemory (object):
    """
    Class for encapsulating and managing feature and kernel matrices for
    different feature types
    """

    @classmethod
    def construct_from_files(cls, id_vector_file, bg_flags_file,
                             feature_mat_file, kernel_mat_file, rw_lock=None):
        """ Initialize FeatureMemory object from file sources.

        :param id_vector_file: File containing the numpy.savetxt(...) output of
            clip ID values in the order in which they associate to the rows of
            the kernel matrix.
        :type id_vector_file: str
        :param feature_mat_file: File containing the kernel matrix as saved by
            numpy.save(...) (saved as an ndarray, converted to matrix on load).
        :type feature_mat_file: str
        :param kernel_mat_file: File containing the kernel matrix as saved by
            numpy.save(...) (saved as an ndarray, converted to matrix on load).
        :type kernel_mat_file: str
        :param bg_flags_file: Optional file containing output of
            numpy.savetxt(...) where each index maps a row index of the kernel
            to whether or not the associated clip ID should be considered a
            background video or not.
        :type bg_flags_file: str
        :return: Symmetric FeatureMemory constructed with the data provided in
            the provided files.
        :rtype: FeatureMemory

        """
        clip_ids = np.array(np.load(id_vector_file))
        bg_flags = np.array(np.load(bg_flags_file))
        # noinspection PyCallingNonCallable
        feature_mat = np.matrix(np.load(feature_mat_file))
        # noinspection PyCallingNonCallable
        kernel_mat = np.matrix(np.load(kernel_mat_file))

        bg_clips = set([clip_ids[i]
                        for i, f in enumerate(bg_flags)
                        if f])

        return FeatureMemory(clip_ids, bg_clips, feature_mat, kernel_mat,
                             rw_lock=rw_lock)

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
        :param bg_clip_ids: Set of clip IDs that are to be treated as background
            clip IDs.
        :type bg_clip_ids: set of int
        :param feature_mat: (numpy) Matrix of features for clip IDs. Features
            should be stored vertically, i.e. Each row is a feature for a
            particular clip ID (id_vector being the index-to-clipID map).
        :type feature_mat: matrix of double
        :param kernel_mat: (numpy) Matrix detailing the distances between
            feature vectors. This must be a square, symmetric matrix.
        :type kernel_mat: matrix of double
        :param rw_lock: Optional ReadWriteLock for this instance to use. If not
            provided, we will create our own.
        :type rw_lock: None or ReadWriteLock

        """
        # assert isinstance(id_vector, (ndarray, ArrayProxy)), \
        #     "ID vector not given as a numpy.ndarray!"
        assert isinstance(bg_clip_ids, (set, frozenset)), \
            "Background ID vector not a numpy.ndarray!"
        # assert isinstance(feature_mat, (matrix, MatrixProxy)), \
        #     "Kernel matrix not a numpy.matrix!"
        # assert isinstance(kernel_mat, (matrix, MatrixProxy)), \
        #     "Distance kernel not a numpy.matrix!"

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

        self._log.debug("Lock given: %s", rw_lock)
        if rw_lock:
            assert isinstance(rw_lock, ReadWriteLock), \
                "Not given a value ReadWriteLock instance!"
            self._rw_lock = rw_lock
        else:
            self._log.debug("Falling back on bad lock given (given: %s)",
                            type(rw_lock))
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
        # noinspection PyUnresolvedReferences
        return (a + b - np.abs(a - b)).sum() * 0.5

    def get_ids(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        with this object's lock when in a parallel environment to prevent
        possible memory corruption.

        :return: Ordered vector of clip IDs along the row-edge of this object's
            feature matrix and along both edges of the kernel matrix.
        :rtype: numpy.core.multiarray.ndarray

        """
        return self._id_vector

    def get_bg_ids(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        with this object's lock when in a parallel environment to prevent
        possible memory corruption.

        :return: Ordered vector of clip IDs that we are treating as background
            clips.
        :rtype: ndarray

        """
        return frozenset(self._bg_clip_ids)

    def get_feature_matrix(self):
        """
        NOTE: NOT THREAD SAFE. Use the returned structure only in conjunction
        with this object's lock when in a parallel environment to prevent
        possible memory corruption.

        :return: Matrix recording feature vectors for a feature type. See the
            id vector for row-wise index-to-clipID association.
        :rtype: numpy.matrixlib.defmatrix.matrix

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
            with SimpleTimer("Allocating return matrix", self._log.debug):
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
    def update(self, clip_id, feature_vec=None, is_background=False, timeout=None):
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
        :param feature_vec: Feature vector associated to the given clip ID.
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
            if self._cid2idx_map.get(clip_id, None) is not None:
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

            # Given a new feature to add.
            else:
                if feature_vec is None:
                    raise ValueError("Update given a new clip ID, but no "
                                     "feature vector provided.")

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
                # noinspection PyUnresolvedReferences
                self._feature_mat.resize((self._feature_mat.shape[0] + 1,
                                          self._feature_mat.shape[1]),
                                         refcheck=False
                                         )
                self._feature_mat[-1, :] = feature_vec

                # Need to add a new row AND column to the distance kernel.
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
        self._map_lock = multiprocessing.RLock()
        #: :type: dict of (str, FeatureMemory)
        self._feature2memory = {}

    def get_feature_types(self):
        """ Get available feature types in this map.

        :return: Tuple of string names of all features initialize in this map
        :rtype: tuple of str

        """
        with self._map_lock:
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
        with self._map_lock:
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
        with self._map_lock:
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
        with self._map_lock:
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
        with self._map_lock:
            return self._feature2memory[feature_type]

    def get_distance_kernel(self, feature_type):
        """ Get the DistanceKernel for a feature type.

        :raise KeyError: If the given feature type does not currently map to
            anything.

        :param feature_type: The feature type to get the memory object of.
        :type feature_type: str

        """
        with self._map_lock:
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
        with self._map_lock:
            return self._feature2memory[feature_type]\
                       .get_feature(*clip_id_or_ids)

    def update(self, feature_type, clip_id, feature_vector, is_background=False,
               timeout=None):
        with self._map_lock:
            return self._feature2memory[feature_type]\
                       .update(clip_id, feature_vector, is_background, timeout)
