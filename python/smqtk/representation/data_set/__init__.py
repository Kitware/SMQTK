import abc
import collections
import os

from smqtk.representation import SmqtkRepresentation
from smqtk.utils import plugin


class DataSet (collections.Set, SmqtkRepresentation, plugin.Pluggable):
    """
    Abstract interface for data sets, that contain an arbitrary number of
    ``DataElement`` instances of arbitrary implementation type, keyed on
    ``DataElement`` UUID values.

    This should only be used with DataElements whose byte content is expected
    not to change. If they do, then UUID keys may no longer represent the
    elements associated with them.

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

    @abc.abstractmethod
    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """

    @abc.abstractmethod
    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        *NOTE: Implementing methods should check that input elements are in
        fact DataElement instances.*

        :param elems: Data element(s) to add
        :type elems: smqtk.representation.DataElement

        """

    @abc.abstractmethod
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


def get_data_set_impls(reload_modules=False):
    """
    Discover and return discovered ``DataSet`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``DATA_SET_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``DATA_SET_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``DataSet``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "DATA_SET_PATH"
    helper_var = "DATA_SET_CLASS"
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var, DataSet,
                              reload_modules=reload_modules)
