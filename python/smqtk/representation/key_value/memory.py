import threading

import six
from six.moves import cPickle as pickle

from smqtk.representation import DataElement, KeyValueStore
from smqtk.representation.key_value import NO_DEFAULT_VALUE
from smqtk.utils.configuration \
    import make_default_config, from_config_dict, to_config_dict


class MemoryKeyValueStore (KeyValueStore):
    """
    Thread-safe in-memory implementation of KeyValueStore interface with
    optional caching from/to a ``DataElement`` instance.

    Any keys and values compatible with a standard python dictionary are
    compatible with this implementation.

    **WARNING:** *This element uses pickle serialization for storing cached
    bytes. This is a security risk when not using secured or authorized storage
    mediums.*

    """

    PICKLE_PROTOCOL = 2

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        default = super(MemoryKeyValueStore, cls).get_default_config()
        default['cache_element'] = make_default_config(DataElement.get_impls())
        return default

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: MemoryKeyValueStore

        """
        # Copy top-level of config in order to not modify input instance.
        c = config_dict.copy()
        # Simplify specification for "no cache element"
        if 'cache_element' not in c or \
                c['cache_element'] is None or \
                c['cache_element']['type'] is None:
            c['cache_element'] = None
        else:
            # Create from nested config.
            c['cache_element'] = \
                from_config_dict(config_dict['cache_element'],
                                 DataElement.get_impls())
        return super(MemoryKeyValueStore, cls).from_config(c)

    def __init__(self, cache_element=None):
        """
        Create new in-memory key-value store with optional cache data element.

        This element is read-only when using a DataElement instance for a cache
        and that element is not writable.

        :param cache_element: Optional data element to load/save state from/to.
        :type cache_element: smqtk.representation.DataElement

        """
        super(MemoryKeyValueStore, self).__init__()
        self._cache_element = cache_element
        self._table = {}
        self._table_lock = threading.RLock()

        # Only try to load from cache if the cache has any bytes to try to
        # deserialize.
        if self._cache_element:
            c_bytes = self._cache_element.get_bytes()
            if c_bytes:
                #: :type: dict
                self._table = pickle.loads(c_bytes)

    def __repr__(self):
        return super(MemoryKeyValueStore, self).__repr__() \
            % ("cache_element: %s" % repr(self._cache_element))

    def cache_table(self):
        """
        Cache the current table to the currently set cache element.

        If there is no cache element, this method does nothing.
        """
        if self._cache_element is not None:
            # TODO(paul.tunison): Some other serialization than Pickle.
            #   - pickle loading allows arbitrary code execution on host.
            self._log.debug("Caching table to {}".format(self._cache_element))
            self._cache_element.set_bytes(
                pickle.dumps(self._table, self.PICKLE_PROTOCOL))

    def count(self):
        with self._table_lock:
            return len(self._table)

    def get_config(self):
        # Recursively get config from data element if we have one.
        if hasattr(self._cache_element, 'get_config'):
            elem_config = to_config_dict(self._cache_element)
        else:
            # No cache element, output default config with no type.
            elem_config = make_default_config(DataElement.get_impls())
        return {
            'cache_element': elem_config
        }

    def keys(self):
        """
        :return: Iterator over keys in this store.
        :rtype: __generator[collections.abc.Hashable]
        """
        return six.iterkeys(self._table)

    def is_read_only(self):
        """
        If this element is read-only or not as determined by any cache element's
        read-only status.

        :return: True if this instance is using a ``DataElement``
        :rtype: bool
        """
        # Only not writable if we have a cache and it is not writable
        if self._cache_element and not self._cache_element.writable():
            return True
        return False

    def has(self, key):
        """
        Check if this store has a value for the given key.

        :param key: Key to check for a value for.
        :type key: collections.abc.Hashable

        :return: If this store has a value for the given key.
        :rtype: bool

        """
        return key in self._table

    def add(self, key, value):
        """
        Add a key-value pair to this store.

        :param key: Key for the value. Must be hashable.
        :type key: collections.abc.Hashable

        :param value: Python object to store.
        :type value: object

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        super(MemoryKeyValueStore, self).add(key, value)
        with self._table_lock:
            self._table[key] = value
            self.cache_table()
        return self

    def add_many(self, d):
        """
        Add multiple key-value pairs at a time into this store as represented in
        the provided dictionary `d`.

        :param d: Dictionary of key-value pairs to add to this store.
        :type d: dict[collections.abc.Hashable, object]

        :return: Self.
        :rtype: MemoryKeyValueStore

        """
        super(MemoryKeyValueStore, self).add_many(d)
        with self._table_lock:
            self._table.update(d)
            self.cache_table()
        return self

    def remove(self, key):
        """
        Remove a single key-value entry.

        :param key: Key to remove.
        :type key: collections.abc.Hashable

        :raises KeyError: The given key is not present in this store and no
            default value given.

        :return: Self.
        :rtype: KeyValueStore

        """
        super(MemoryKeyValueStore, self).remove(key)
        with self._table_lock:
            del self._table[key]
            self.cache_table()
        return self

    def remove_many(self, keys):
        """
        Remove multiple keys and associated values.

        :param keys: Iterable of keys to remove.  If this is empty this method
            does nothing.
        :type keys: collections.abc.Iterable[collections.abc.Hashable]

        :raises KeyError: The given key is not present in this store and no
            default value given.  The store is not modified if any key is
            invalid.

        :return: Self.
        :rtype: KeyValueStore

        """
        super(MemoryKeyValueStore, self).remove_many(keys)
        with self._table_lock:
            # Make sure all keys are represented
            key_diff = set(keys) - set(self._table)
            if key_diff:
                if len(key_diff) == 1:
                    raise KeyError(list(key_diff)[0])
                else:
                    raise KeyError(key_diff)
            # Actually remove keys.
            for k in keys:
                del self._table[k]
            self.cache_table()
        return self

    def get(self, key, default=NO_DEFAULT_VALUE):
        """
        Get the value for the given key.

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
        with self._table_lock:
            if default is NO_DEFAULT_VALUE:
                return self._table[key]
            else:
                return self._table.get(key, default)

    def clear(self):
        """
        Clear this key-value store.

        :return: Self.
        :rtype: KeyValueStore

        """
        super(MemoryKeyValueStore, self).clear()
        with self._table_lock:
            self._table.clear()
        return self
