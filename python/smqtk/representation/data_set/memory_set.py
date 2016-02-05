import cPickle
import multiprocessing
import os

from smqtk.representation import DataElement, DataSet
from smqtk.utils import SimpleTimer


__author__ = 'paul.tunison@kitware.com'


class DataMemorySet (DataSet):
    """
    In-memory DataSet implementation. This does not support a persistant
    representation.
    """

    @classmethod
    def is_usable(cls):
        """
        Check whether this data set implementations is available for use.

        This is always true for this implementation as there are no required 3rd
        party dependencies

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        return True

    def __init__(self, file_cache=None, pickle_protocol=-1):
        """
        Initialize a new in-memory data set instance.

        :param file_cache: Optional path to a file to store/load a cache of this
            data set's contents into. Cache loading, if the file was found, will
            occur in this constructor. Cache writing will only occur after
            adding one or more elements.

            This can be optionally turned on after creating/using this data set
            for a while by setting a valid filepath to the ``file_cache``
            attribute and calling the ``.cache()`` method. When ``file_cache``
            is not set, the ``cache()`` method does nothing.
        :type file_cache: None | str

        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :type pickle_protocol: int

        """
        super(DataMemorySet, self).__init__()

        # Mapping of UUIDs to DataElement instances
        #: :type: dict[collections.Hashable, DataElement]
        self._element_map = {}
        self._element_map_lock = multiprocessing.RLock()

        # Optional path to a file that will act as a cache of our internal
        # table
        self.file_cache = file_cache
        if file_cache and os.path.isfile(file_cache):
            with open(file_cache) as f:
                #: :type: dict[collections.Hashable, DataElement]
                self._element_map = cPickle.load(f)

        self.pickle_protocol = pickle_protocol

    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in no
            particular order.
        """
        # making copy of UUIDs so we don't block when between yields, as well
        # as so we aren't walking a possibly modified map
        uuids = self.uuids()
        with self._element_map_lock:
            for k in uuids:
                yield self._element_map[k]

    def cache(self):
        if self.file_cache:
            with self._element_map_lock:
                with SimpleTimer("Caching memory data-set table", self._log.info):
                    with open(self.file_cache, 'wb') as f:
                        cPickle.dump(self._element_map, f,
                                     self.pickle_protocol)

    def get_config(self):
        """
        This implementation has no configuration properties.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            "file_cache": self.file_cache,
            "pickle_protocol": self.pickle_protocol,
        }

    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        with self._element_map_lock:
            return len(self._element_map)

    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """
        with self._element_map_lock:
            return set(self._element_map)

    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the
            type that the set implementation expects or cares about.

        :return: True if the given uuid matches an element in this set, or
            False if it does not.
        :rtype: bool

        """
        with self._element_map_lock:
            return uuid in self._element_map

    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        :param elems: Data element(s) to add
        :type elems: list[smqtk.representation.DataElement]

        """
        with self._element_map_lock:
            for e in elems:
                assert isinstance(e, DataElement), "Expected DataElement instance, got '%s' instance instead" % type(e)
                self._element_map[e.uuid()] = e
            self.cache()

    def get_data(self, uuid):
        """
        Get the data element the given uuid references, or raise an
        exception if the uuid does not reference any element in this set.

        :raises KeyError: If the given uuid does not refer to an element in
            this data set.

        :param uuid: The uuid of the element to retrieve.

        :return: The data element instance for the given uuid.
        :rtype: smqtk.representation.DataElement

        """
        with self._element_map_lock:
            return self._element_map[uuid]


DATA_SET_CLASS = DataMemorySet
