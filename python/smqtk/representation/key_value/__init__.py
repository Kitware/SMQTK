"""
Data abstraction interface for general key-value storage.
"""
import abc

import six

from smqtk.exceptions import ReadOnlyError
from smqtk.representation import SmqtkRepresentation
from smqtk.utils.plugin import Pluggable


NO_DEFAULT_VALUE = type("KeyValueStoreNoDefaultValueType", (object,), {})()


class KeyValueStore (SmqtkRepresentation, Pluggable):
    """
    Interface for general string key, python object storage.

    Objects are serialized via the ``pickle`` module, thus input objects must be
    picklable.
    """

    # Mutable storage container is not hashable.
    __hash__ = None

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
    def is_read_only(self):
        """
        :return: True if this instance is read-only and False if it is not.
        :rtype: bool
        """

    @abc.abstractmethod
    def add(self, key, value):
        """
        Add a key-value pair to this store.

        *NOTE:* **Implementing sub-classes should call this super-method.**

        :param key: String key for the value.
        :type key: str

        :param value: Python object to store. This must be picklable.
        :type value: object

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        if not isinstance(key, six.string_types):
            raise ValueError("Key is not a string type.")
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

    @abc.abstractmethod
    def has(self, key):
        """
        Check if this store has a value for the given key.

        :param key: String key to check for a value for.
        :type key: str

        :return: If this store has a value for the given key.
        :rtype: bool

        """

    @abc.abstractmethod
    def get(self, key, default=NO_DEFAULT_VALUE):
        """
        Get the value for the given string key.

        *NOTE:* **Implementing sub-classes are responsible for raising a
        ``KeyError`` where appropriate.**

        :param key: String key to get the value of.
        :type key: str

        :param default: Optional default value if the given key is not present
            in this store. This may be any value except for the
            ``NO_DEFAULT_VALUE`` constant (custom anonymous class instance).
        :type default: object

        :raises KeyError: The given key is not present in this store and no
            default value given.

        :return: Deserialized python object stored for the given key.
        :rtype: object

        """
