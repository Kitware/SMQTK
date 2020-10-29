"""
Data abstraction interface for general key-value storage.
"""
import abc
import collections  # noqa: F401

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
    __hash__ = None  # type: ignore

    def __len__(self):
        return self.count()

    def __contains__(self, item):
        return self.has(item)

    def __getitem__(self, item):
        return self.get(item)

    @abc.abstractmethod
    def __repr__(self):
        """
        Return representative string for this class.

        *NOTE:* **This abstract super-method returns a template string to add
        to sub-class specific information to. The returned string should be
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
        :rtype: collections.abc.Iterator[collections.abc.Hashable]
        """

    def values(self):
        """
        :return: Iterator over values in this store. Values are not guaranteed
            to be in any particular order.
        :rtype: collections.abc.Iterator
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
        :type key: collections.abc.Hashable

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
        :type key: collections.abc.Hashable

        :param value: Python object to store.
        :type value: object

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

    @abc.abstractmethod
    def add_many(self, d):
        """
        Add multiple key-value pairs at a time into this store as represented
        in the provided dictionary `d`.

        :param d: Dictionary of key-value pairs to add to this store.
        :type d: dict[collections.abc.Hashable, object]

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        # Input keys must already be hashable because they're in a dictionary.
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

    @abc.abstractmethod
    def remove(self, key):
        """
        Remove a single key-value entry.

        :param key: Key to remove.
        :type key: collections.abc.Hashable

        :raises ReadOnlyError: If this instance is marked as read-only.
        :raises KeyError: The given key is not present in this store and no
            default value given.

        :return: Self.
        :rtype: KeyValueStore

        """
        if self.is_read_only():
            raise ReadOnlyError("Cannot remove from read-only instance %s."
                                % self)

    @abc.abstractmethod
    def remove_many(self, keys):
        """
        Remove multiple keys and associated values.

        :param keys: Iterable of keys to remove.  If this is empty this method
            does nothing.
        :type keys: collections.abc.Iterable[collections.abc.Hashable]

        :raises ReadOnlyError: If this instance is marked as read-only.
        :raises KeyError: The given key is not present in this store and no
            default value given.  The store is not modified if any key is
            invalid.

        :return: Self.
        :rtype: KeyValueStore

        """
        if self.is_read_only():
            raise ReadOnlyError("Cannot remove from read-only instance %s."
                                % self)

    @abc.abstractmethod
    def get(self, key, default=NO_DEFAULT_VALUE):
        """
        Get the value for the given key.

        *NOTE:* **Implementing sub-classes are responsible for raising a
        ``KeyError`` where appropriate.**

        :param key: Key to get the value of.
        :type key: collections.abc.Hashable

        :param default: Optional default value if the given key is not present
            in this store. This may be any value except for the
            ``NO_DEFAULT_VALUE`` constant (custom anonymous class instance).
        :type default: object

        :raises KeyError: The given key is not present in this store and no
            default value given.

        :return: Deserialized python object stored for the given key.
        :rtype: object

        """

    def get_many(self, keys, default=NO_DEFAULT_VALUE):
        """
        Get the values for the given keys.

        *NOTE:* **Implementing sub-classes are responsible for raising a
        ``KeyError`` where appropriate.**

        :param keys: The keys for which associated values are requested.
        :type keys: collections.abc.Iterable[collections.abc.Hashable]

        :param default: Optional default value if a given key is not present
            in this store. This may be any value except for the
            ``NO_DEFAULT_VALUE`` constant (custom anonymous class instance).
        :type default: object

        :raises KeyError: A given key is not present in this store and no
            default value given.

        :return: Iterable of deserialized python objects stored for the given
            keys in the order that the corresponding keys were provided.
        :rtype: collections.abc.Iterable

        """
        for key_ in keys:
            yield self.get(key_, default=default)

    @abc.abstractmethod
    def clear(self):
        """
        Clear this key-value store.

        *NOTE:* **Implementing sub-classes should call this super-method. This
        super method should not be considered a critical section for thread
        safety.**

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        if self.is_read_only():
            raise ReadOnlyError("Cannot clear a read-only %s instance."
                                % self.__class__.__name__)
