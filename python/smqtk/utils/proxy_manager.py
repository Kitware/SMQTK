"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""
# -*- coding: utf-8 -*-
from multiprocessing import current_process
from multiprocessing.managers import (
    all_methods,
    BaseProxy,
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

from .distance_kernel import DistanceKernel
from .feature_memory import FeatureMemory, FeatureMemoryMap
from . import ReadWriteLock
from .timed_cache import TimedCache


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
# TODO: Check out Pyro as an alternative to multiprocessing.managers

class BaseProxy2 (BaseProxy):
    """
    Intermediate class under BaseProxy in order to fix the resetting of the
    _manager on fork (doesn't need to be? security issues?).
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
# TODO: Add``method_to_typeid`` for numpy.array functions that return copies of data
class ArrayProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(ndarray)
    _exposed_properties_ = all_safe_properties(ndarray)
    _method_to_typeid_ = {
        '__iter__': "Iterator"
    }


# Wrapper around numpy matrix class, providing pass-through functions for class
# functions as of numpy 1.8.0.
# TODO: Add``method_to_typeid`` for numpy.matrix functions that return copies of data
class MatrixProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(matrix)
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

# Return types can't seem to be the same as a type that is creatable, so we use
# a standard 'r' character prefix on return types. This required a manager
# register call below with that name with the option ``create_method=False``.

class DistanceKernelProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(DistanceKernel)
    _method_to_typeid_ = {
        'get_lock':             'rReadWriteLock',
        'row_id_map':           'rarray',
        'col_id_map':           'rarray',
        'get_kernel_matrix':    'rmatrix',
        'get_background_ids':   'rarray'
    }


class FeatureMemoryProxy (BaseProxy2):
    __metaclass__ = ExposedAutoGenMeta
    _exposed_ = all_safe_methods(FeatureMemory)
    _method_to_typeid_ = {
        '_generate_distance_kernel_matrix': 'rmatrix',

        'get_ids':              'rarray',
        'get_bg_ids':           'rarray',
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
# ProxyManager and Type registration
#
class ProxyManager (SyncManager):
    """
    This class shouldn't be initialized directly, but instead initialized and
    retrieved through the init and get functions below.
    """
    pass


# Object-based
ProxyManager.register('array',            np.array,         ArrayProxy)
ProxyManager.register('ndarray',          np.ndarray,       ArrayProxy)
ProxyManager.register('matrix',           np.matrix,        MatrixProxy)
ProxyManager.register('ReadWriteLock',    ReadWriteLock,    ReadWriteLockProxy)
ProxyManager.register('DistanceKernel',   DistanceKernel,   DistanceKernelProxy)
ProxyManager.register('FeatureMemory',    FeatureMemory,    FeatureMemoryProxy)
ProxyManager.register('FeatureMemoryMap', FeatureMemoryMap, FeatureMemoryMapProxy)
# Function based
ProxyManager.register('get_common_fmm',   get_common_fmm,   FeatureMemoryMapProxy)
ProxyManager.register('get_common_tc',    get_common_tc,    TimedCacheProxy)
ProxyManager.register("symmetric_dk_from_file",
                      DistanceKernel.construct_symmetric_from_files,
                      DistanceKernelProxy)
ProxyManager.register("asymmetric_dk_from_file",
                      DistanceKernel.construct_asymmetric_from_files,
                      DistanceKernelProxy)

# Return proxy registrations
# - If a function Proxy has a method that should return another proxy, the
#   proxy type that it returns CANNOT have been registered with
#   ``create_method=True``. Reason: the server uses the registered typeid's
#   callable to create the return object
ProxyManager.register('rarray',           proxytype=ArrayProxy,           create_method=False)
ProxyManager.register("rmatrix",          proxytype=MatrixProxy,          create_method=False)
# Generated structure from read_lock() and write_lock() methods in ReadWriteLock
ProxyManager.register("rRWLockWith",      proxytype=RWLockWithProxy,      create_method=False)
ProxyManager.register("rReadWriteLock",   proxytype=ReadWriteLockProxy,   create_method=False)
ProxyManager.register("rDistanceKernel",  proxytype=DistanceKernelProxy,  create_method=False)
ProxyManager.register("rFeatureMemory",   proxytype=FeatureMemoryProxy,   create_method=False)


# Map of ProxyManager instance keyed on the address
#: :type: dict of (tuple or str or None, ProxyManager)
__mgr_cache__ = {}


# noinspection PyPep8Naming
def initFeatureManagerConnection(address=None, authkey=None,
                                 serializer='pickle'):
    """ Initialize ProxyManager connection

    :raises ValueError: The given address already maps to an initialized
        ProxyManager.

    :param address: The address of the ProxyManager server to connect to, or
        None for the local process tree's ProxyManager. When None, the server
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
                         "initialized ProxyManager!"
                         % str(address))
    # Nope, I'm pretty sure that constructor exists in the base class...
    # noinspection PyArgumentList
    __mgr_cache__[address] = ProxyManager(address, authkey, serializer)

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
    """ Shutdown and remove an initialized ProxyManager connection
    :raises KeyError: if the given address is not associated with to an active
        manager.
    :param address: The address of the ProxyManager connection to remove from
        the active mapping.
    :type address: str or (str, int) or None
    """
    global __mgr_cache__
    #: :type: ProxyManager
    if hasattr(__mgr_cache__[address], 'shutdown'):
        __mgr_cache__[address].shutdown()
    del __mgr_cache__[address]


# noinspection PyPep8Naming
def getFeatureManager(address=None):
    """
    Get the ProxyManager instance for the given address from the cache. If the
    address is None, returns the ProxyManager for the current process tree
    (may have been initialized on a parent process).

    :raises KeyError: When the given address has not been initialized and is not
        present in the cache.

    :param address: The address of the ProxyManager connection to retrieve
        from the active mapping.
    :type address: str or (str, int) or None

    :return: Singleton feature_manager for the given address.
    :rtype: ProxyManager

    """
    return __mgr_cache__[address]
