import six

from smqtk.representation import DescriptorIndex, get_data_element_impls
from smqtk.utils import merge_dict, plugin, SimpleTimer

try:
    import cPickle as pickle
except ImportError:
    import pickle


class MemoryDescriptorIndex (DescriptorIndex):
    """
    In-memory descriptor index with file caching.

    Stored descriptor elements are all held in memory in a uuid-to-element
    dictionary (hash table).

    If the path to a file cache is provided, it is loaded at construction if it
    exists. When elements are added to the index, the in-memory table is dumped
    to the cache.
    """

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        # no dependencies
        return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as arguments,
        turning those argument names into configuration dictionary keys. If any
        of those arguments have defaults, we will add those values into the
        configuration dictionary appropriately. The dictionary returned should
        only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        c = super(MemoryDescriptorIndex, cls).get_default_config()
        c['cache_element'] = plugin.make_config(get_data_element_impls())
        return c

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
        :rtype: MemoryDescriptorIndex

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        # Optionally construct cache element from sub-config.
        if config_dict['cache_element'] \
                and config_dict['cache_element']['type']:
            e = plugin.from_plugin_config(config_dict['cache_element'],
                                          get_data_element_impls())
            config_dict['cache_element'] = e
        else:
            config_dict['cache_element'] = None

        return super(MemoryDescriptorIndex, cls).from_config(config_dict, False)

    def __init__(self, cache_element=None, pickle_protocol=-1):
        """
        Initialize a new in-memory descriptor index, or reload one from a
        cache.

        :param cache_element: Optional data element cache, loading an existing
            index if the element has bytes. If the given element is writable,
             new descriptors added to this index are cached to the element.
        :type cache_element: None | smqtk.representation.DataElement

        :param pickle_protocol: Pickling protocol to use when serializing index
            table to the optionally provided, writable cache element. We will
            use -1 by default (latest version, probably a binary form).
        :type pickle_protocol: int

        """
        super(MemoryDescriptorIndex, self).__init__()

        # Mapping of descriptor UUID to the DescriptorElement instance.
        #: :type: dict[collections.Hashable, smqtk.representation.DescriptorElement]
        self._table = {}
        # Record of optional file cache we're using
        self.cache_element = cache_element
        self.pickle_protocol = pickle_protocol

        if cache_element and not cache_element.is_empty():
            self._log.debug("Loading cached descriptor index table from %s "
                            "element.", cache_element.__class__.__name__)
            self._table = pickle.loads(cache_element.get_bytes())

    def get_config(self):
        c = merge_dict(self.get_default_config(), {
            "pickle_protocol": self.pickle_protocol,
        })
        if self.cache_element:
            merge_dict(c['cache_element'],
                       plugin.to_plugin_config(self.cache_element))
        return c

    def cache_table(self):
        if self.cache_element and self.cache_element.writable():
            with SimpleTimer("Caching descriptor table", self._log.debug):
                self.cache_element.set_bytes(pickle.dumps(self._table,
                                                          self.pickle_protocol))

    def count(self):
        return len(self._table)

    def clear(self):
        """
        Clear this descriptor index's entries.
        """
        self._table = {}
        self.cache_table()

    def has_descriptor(self, uuid):
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for
        :type uuid: collections.Hashable

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.
        :rtype: bool

        """
        return uuid in self._table

    def add_descriptor(self, descriptor, no_cache=False):
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple
        copies of the descriptor in the index.

        :param descriptor: Descriptor to index.
        :type descriptor: smqtk.representation.DescriptorElement

        :param no_cache: Do not cache the internal table if a file cache was
            provided. This would be used if adding many descriptors at a time,
            preventing a file write for every individual descriptor added.
        :type no_cache: bool

        """
        self._table[descriptor.uuid()] = descriptor
        if not no_cache:
            self.cache_table()

    def add_many_descriptors(self, descriptors):
        """
        Add multiple descriptors at one time.

        :param descriptors: Iterable of descriptor instances to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        added_something = False
        for d in descriptors:
            # using no-cache so we don't trigger multiple file writes
            self.add_descriptor(d, no_cache=True)
            added_something = True
        if added_something:
            self.cache_table()

    def get_descriptor(self, uuid):
        """
        Get the descriptor in this index that is associated with the given UUID.

        :param uuid: UUID of the DescriptorElement to get.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        :return: DescriptorElement associated with the queried UUID.
        :rtype: smqtk.representation.DescriptorElement

        """
        return self._table[uuid]

    def get_many_descriptors(self, uuids):
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated to given uuid values.
        :rtype: __generator[smqtk.representation.DescriptorElement]

        """
        for uid in uuids:
            yield self._table[uid]

    def remove_descriptor(self, uuid, no_cache=False):
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        :param no_cache: Do not cache the internal table if a file cache was
            provided. This would be used if adding many descriptors at a time,
            preventing a file write for every individual descriptor added.
        :type no_cache: bool

        """
        del self._table[uuid]
        if not no_cache:
            self.cache_table()

    def remove_many_descriptors(self, uuids):
        """
        Remove descriptors associated to given descriptor UUIDs from this
        index.

        :param uuids: Iterable of descriptor UUIDs to remove.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """
        for uid in uuids:
            # using no-cache so we don't trigger multiple file writes
            self.remove_descriptor(uid, no_cache=True)
        self.cache_table()

    def iterkeys(self):
        return six.iterkeys(self._table)

    def iterdescriptors(self):
        return six.itervalues(self._table)

    def iteritems(self):
        return six.iteritems(self._table)


DESCRIPTOR_INDEX_CLASS = MemoryDescriptorIndex
