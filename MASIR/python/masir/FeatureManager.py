"""

Extension of the multiprocessing.managers module, which is limited in is
convenient use

"""

import multiprocessing.managers
import multiprocessing.util


class BaseProxy2 (multiprocessing.managers.BaseProxy):
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
        ret[1][-1]['manager'] = self._manager_copy
        return ret


#
# Helper methods for custom Proxy creation
#

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
    return tuple(set(multiprocessing.managers.all_methods(obj))
                 .difference(__leave_alone_methods))


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
    """
    Metaclass for convenient creation of custom class proxies

    class variables that affect proxy creation:

        "_exposed_"
            Iterable of class method names to expose from the underlying class.
            See convenience function "all_safe_methods(...)"

        "_exposed_properties_"
            Iterable of class property names to expose from the underlying
            class. See convenience function "all_safe_properties(...)"

        "_method_to_typeid_"
            Dictionary mapping the names of functions to the proxy type that
            should be used to wrap return values coming from the manager server.
            I.e. when a given method is called on a proxy, a proxy is returned
            instead of a copy of the return data.

    """

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


#
# Shell around SyncManager for future customization
#

class FeatureManager (multiprocessing.managers.SyncManager):
    """
    This class shouldn't be initialized directly, but instead initialized and
    retrieved through the init and get functions below.

    Only one instance of this class should be constructed per process (appends
    to current_process() object)

    """

    # List of registered type IDs to check for null registering (improper use)
    _nondefault_proxy_types = set()

    @classmethod
    def register(cls, typeid, callable=None, proxytype=None, exposed=None,
                 method_to_typeid=None, create_method=True):
        super(FeatureManager, cls).register(typeid, callable, proxytype,
                                            exposed, method_to_typeid,
                                            create_method)
        if '__REGISTERED_PROXIES' not in cls.__dict__:
            cls._nondefault_proxy_types = cls._nondefault_proxy_types.copy()
        cls._nondefault_proxy_types.add(typeid)


#
# State-based FeatureManager connection and accessor functions
#

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

    # Check that the FeatureManager class has any new proxies registered. If
    # not, the user probably forgot to access this init function after passing
    # through a secondary file that defines proxy registration.
    # noinspection PyProtectedMember
    if not FeatureManager._nondefault_proxy_types:
        raise RuntimeError("Failed to initialize FeatureManager due to it only "
                           "containing default registrants (SyncManager). Are "
                           "you sure you imported the proxy registration file "
                           "first?")

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