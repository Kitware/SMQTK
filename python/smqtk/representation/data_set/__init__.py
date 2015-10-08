import abc
import collections

from smqtk.representation import SmqtkRepresentation
from smqtk.utils import plugin


__author__ = "paul.tunison@kitware.com"


class DataSet (collections.Set, SmqtkRepresentation, plugin.Pluggable):
    """
    Abstract interface for data sets, that contain an arbitrary number of
    ``DataElement`` instances of arbitrary implementation type, keyed on
    ``DataElement`` UUID values.

    """

    @classmethod
    @abc.abstractmethod
    def is_usable(cls):
        """
        Check whether this data set implementations is available for use.

        Since certain implementations may require additional dependencies that
        may not yet be available on the system, this method should check for
        those dependencies and return a boolean saying if the implementation is
        usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should emit a
                warning detailing why the implementation is not available for
                use.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """

    def __len__(self):
        """
        :return: Number of elements in this DataSet.
        :rtype: int
        """
        return self.count()

    def __getitem__(self, uuid):
        return self.get_data(uuid)

    def __contains__(self, d):
        """
        Different than has_uuid() because this takes another DataElement
        instance, not a UUID.

        :param d: DataElement to test for containment
        :type d: smqtk.representation.DataElement

        :return: True of this DataSet contains the given data element. Since,
        :rtype: bool

        """
        return self.has_uuid(d.uuid())

    @abc.abstractmethod
    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in no
            particular order.
        """
        return

    @abc.abstractmethod
    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        return

    @abc.abstractmethod
    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """
        return

    @abc.abstractmethod
    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the
            type that the set implementation expects or cares about.

        :return: True if the given uuid matches an element in this set, or
            False if it does not.
        :rtype: bool

        """
        return

    @abc.abstractmethod
    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        :param elems: Data element(s) to add
        :type elems: list[smqtk.representation.DataElement]

        """
        return

    @abc.abstractmethod
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
        return


def get_data_set_impls(reload_modules=False):
    """
    Discover and return DataSet implementation classes found in the plugin
    directory. Keys in the returned map are the names of the discovered classes
    and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with and alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module, we first look for a helper variable by the name
    ``DATA_SET_CLASS``, which can either be a single class object or an
    iterable of class objects, to be exported. If the variable is set to None,
    we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same na e and casing as the module's
    name. If neither are found, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class objects of type ``DataSet`` whose keys
        are the string names of the classes.
    :rtype: dict[str, type]

    """
    import os
    from smqtk.utils.plugin import get_plugins

    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "DATA_SET_CLASS"
    return get_plugins(__name__, this_dir, helper_var, DataSet, reload_modules)
