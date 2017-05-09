from smqtk.utils import merge_dict
from smqtk.utils import plugin
from smqtk.representation import (
    DataElement,
    DataSet,
    KeyValueStore,
    get_key_value_store_impls
)
from smqtk.representation.key_value.memory import MemoryKeyValueStore


DFLT_KVSTORE = MemoryKeyValueStore()


class KVSDataSet (DataSet):
    """
    DataSet backed by a KeyValueStore implementation.

    Since KeyValue stores should be able to contain arbitrary hashable keys and
    arbitrary values, this leverages available implementations for the
    KeyValueStore interface.
    """

    @classmethod
    def is_usable(cls):
        """
        This implementation is always usable.
        :return: True
        :rtype: bool
        """
        # No external dependencies.
        return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict
        """
        c = super(KVSDataSet, cls).get_default_config()
        c['kvstore'] = merge_dict(
            plugin.make_config(get_key_value_store_impls()),
            plugin.to_plugin_config(c['kvstore'])
        )
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
        :rtype: KVSDataSet
        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        # Convert KVStore config to instance for constructor.
        kvs_inst = plugin.from_plugin_config(config_dict['kvstore'],
                                             get_key_value_store_impls())
        config_dict['kvstore'] = kvs_inst

        return super(KVSDataSet, cls).from_config(config_dict, False)

    def __init__(self, kvstore=DFLT_KVSTORE):
        """
        Create new instance.

        If no key-value store is provided, and empty in-memory implementation
        instance is used.

        :param kvstore: Backing key-value store instance.
        :type kvstore: smqtk.representation.KeyValueStore
        """
        super(KVSDataSet, self).__init__()
        assert isinstance(kvstore, KeyValueStore), \
            "Not constructed with a KeyValueStore instance."
        self._kvstore = kvstore

    def get_config(self):
        return {
            'kvstore': plugin.to_plugin_config(self._kvstore)
        }

    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in no
            particular order.
        """
        return self._kvstore.keys()

    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        return len(self._kvstore)

    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """
        return set(self._kvstore.keys())

    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the
            type that the set implementation expects or cares about.
        :type uuid: collections.Hashable

        :return: True if the given uuid matches an element in this set, or
            False if it does not.
        :rtype: bool
        """
        return self._kvstore.has(uuid)

    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        :param elems: Data element(s) to add
        :type elems: smqtk.representation.DataElement
        """
        d = {}
        for e in elems:
            if isinstance(e, DataElement):
                d[e.uuid()] = e
            else:
                raise ValueError("Invalid element '%s'" % e)
        self._kvstore.add_many(d)

    def get_data(self, uuid):
        """
        Get the data element the given uuid references, or raise an
        exception if the uuid does not reference any element in this set.

        :raises KeyError: If the given uuid does not refer to an element in
            this data set.

        :param uuid: The uuid of the element to retrieve.
        :type uuid: collections.Hashable

        :return: The data element instance for the given uuid.
        :rtype: smqtk.representation.DataElement
        """
        return self._kvstore.get(uuid)
