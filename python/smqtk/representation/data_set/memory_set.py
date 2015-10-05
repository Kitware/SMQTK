import multiprocessing

from smqtk.representation import DataElement, DataSet


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

    def __init__(self):
        """
        Initialize a new in-memory data set instance.
        """
        # Mapping of UUIDs to DataElement instances
        #: :type: dict[collections.Hashable, DataElement]
        self._element_map = {}
        self._element_map_lock = multiprocessing.RLock()

    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in no
            particular order.
        """
        # making copy of UUIDs so we don't block when between yields, as well as
        # so we aren't walking a possibly modified map
        uuids = self.uuids()
        with self._element_map_lock:
            for k in uuids:
                yield self._element_map[k]

    def get_config(self):
        """
        This implementation has no configuration properties.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {}

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
                assert isinstance(e, DataElement)
                self._element_map[e.uuid()] = e

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
