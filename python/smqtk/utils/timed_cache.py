"""
LICENCE
-------
Copyright 2013-2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

import logging
import multiprocessing
import time

from smqtk.utils.distance_kernel import DistanceKernel
from smqtk.utils.feature_memory import FeatureMemory


class TimedCache (object):
    """
    Be able to generally store FeatureMemory/DistanceKernel objects with timed
    expiry / deletion.
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
        General getter function. When used with ProxyManager, I think this
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
