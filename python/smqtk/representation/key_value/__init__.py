"""
Data abstraction interface for general key-value storage.
"""
import abc
import collections
import os

from smqtk.exceptions import ReadOnlyError
from smqtk.representation import SmqtkRepresentation
from smqtk.utils.plugin import Pluggable


NO_DEFAULT_VALUE = type("KeyValueStoreNoDefaultValueType", (object,), {})()


class KeyValueStore (SmqtkRepresentation, Pluggable):
    """
    Interface for general key/value storage.

    Implementations may impose restrictions on what types keys or values may be
    due to backend used.

    Data access and manipulation should be thread-safe.
    """

    # Mutable storage container is not hashable.
    __hash__ = None

    def __len__(self):
        return self.count()

    def __contains__(self, item):
        return self.has(item)

    @abc.abstractmethod
    def __repr__(self):
        """
        Return representative string for this class.

        *NOTE:* **This abstract super-method returns a template string to add to
        sub-class specific information to. The returned string should be
        formatted using the ``%`` operator and expects a single string
        argument.**

        :return: Representative string for this class.
        :rtype: str

        """
        return '<' + self.__class__.__name__ + " %s>"

    @abc.abstractmethod
    def count(self):
        """
        :return: The number of key-value relationships in this store.
        :rtype: int | long
        """

    @abc.abstractmethod
    def keys(self):
        """
        :return: Iterator over keys in this store.
        :rtype: collections.Iterator[collections.Hashable]
        """

    def values(self):
        """
        :return: Iterator over values in this store. Values are not guaranteed
            to be in any particular order.
        :rtype: collections.Iterator[object]
        """
        for k in self.keys():
            yield self.get(k)

    @abc.abstractmethod
    def is_read_only(self):
        """
        :return: True if this instance is read-only and False if it is not.
        :rtype: bool
        """

    @abc.abstractmethod
    def has(self, key):
        """
        Check if this store has a value for the given key.

        :param key: Key to check for a value for.
        :type key: collections.Hashable

        :return: If this store has a value for the given key.
        :rtype: bool

        """

    @abc.abstractmethod
    def add(self, key, value):
        """
        Add a key-value pair to this store.

        *NOTE:* **Implementing sub-classes should call this super-method. This
        super method should not be considered a critical section for thread
        safety unless ``is_read_only`` is not thread-safe.**

        :param key: Key for the value. Must be hashable.
        :type key: collections.Hashable

        :param value: Python object to store.
        :type value: object

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        if not isinstance(key, collections.Hashable):
            raise ValueError("Key is not a hashable type.")
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

    @abc.abstractmethod
    def add_many(self, d):
        """
        Add multiple key-value pairs at a time into this store as represented in
        the provided dictionary `d`.

        :param d: Dictionary of key-value pairs to add to this store.
        :type d: dict[collections.Hashable, object]

        :return: Self.
        :rtype: KeyValueStore

        """

    @abc.abstractmethod
    def get(self, key, default=NO_DEFAULT_VALUE):
        """
        Get the value for the given key.

        *NOTE:* **Implementing sub-classes are responsible for raising a
        ``KeyError`` where appropriate.**

        :param key: Key to get the value of.
        :type key: collections.Hashable

        :param default: Optional default value if the given key is not present
            in this store. This may be any value except for the
            ``NO_DEFAULT_VALUE`` constant (custom anonymous class instance).
        :type default: object

        :raises KeyError: The given key is not present in this store and no
            default value given.

        :return: Deserialized python object stored for the given key.
        :rtype: object

        """

    # TODO: get_many(self, keys, default=NO_DEFAULT_VALUE)

    @abc.abstractmethod
    def clear(self):
        """
        Clear this key-value store.

        *NOTE:* **Implementing sub-classes should call this super-method. This
        super method should not be considered a critical section for thread
        safety.**

        :raises ReadOnlyError: If this instance is marked as read-only.

        """
        if self.is_read_only():
            raise ReadOnlyError("Cannot clear a read-only %s instance."
                                % self.__class__.__name__)


def get_key_value_store_impls(reload_modules=False):
    """
    Discover and return discovered ``KeyValueStore`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable
          ``KEY_VALUE_STORE_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``KEY_VALUE_STORE_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``KeyValueStore``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    from smqtk.utils.plugin import get_plugins
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "KEY_VALUE_STORE_PATH"
    helper_var = "KEY_VALUE_STORE_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var,
                       KeyValueStore, reload_modules=reload_modules)
