import abc
import os.path as osp

from smqtk.representation import SmqtkRepresentation
from smqtk.utils import plugin


__author__ = 'paul.tunison@kitware.com'


class DescriptorIndex (SmqtkRepresentation, plugin.Pluggable):
    """
    Index of descriptors, keyed and query-able by descriptor UUID.

    Note that these indexes do not use the descriptor type strings. Thus, if
    a set of descriptors has multiple elements with the same UUID, but
    different type strings, they will bash each other in these indexes. In such
    a case, when dealing with descriptors for different generators, it is
    advisable to use multiple indices.

    """

    def __delitem__(self, uuid):
        self.remove_descriptor(uuid)

    def __getitem__(self, uuid):
        return self.get_descriptor(uuid)

    def __iter__(self):
        return self.iterdescriptors()

    def __len__(self):
        return self.count()

    @abc.abstractmethod
    def count(self):
        """
        :return: Number of descriptor elements stored in this index.
        :rtype: int
        """

    @abc.abstractmethod
    def clear(self):
        """
        Clear this descriptor index's entries.
        """

    @abc.abstractmethod
    def has_descriptor(self, uuid):
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for
        :type uuid: collections.Hashable

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.
        :rtype: bool

        """

    @abc.abstractmethod
    def add_descriptor(self, descriptor):
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptor: Descriptor to index.
        :type descriptor: smqtk.representation.DescriptorElement

        """

    @abc.abstractmethod
    def add_many_descriptors(self, descriptors):
        """
        Add multiple descriptors at one time.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptors: Iterable of descriptor instances to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
    def remove_descriptor(self, uuid):
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        """

    @abc.abstractmethod
    def remove_many_descriptors(self, uuids):
        """
        Remove descriptors associated to given descriptor UUIDs from this index.

        :param uuids: Iterable of descriptor UUIDs to remove.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """

    @abc.abstractmethod
    def iterkeys(self):
        """
        Return an iterator over indexed descriptor keys, which are their UUIDs.
        :rtype: collections.Iterator[collections.Hashable]
        """

    @abc.abstractmethod
    def iterdescriptors(self):
        """
        Return an iterator over indexed descriptor element instances.
        :rtype: collections.Iterator[smqtk.representation.DescriptorElement]
        """

    @abc.abstractmethod
    def iteritems(self):
        """
        Return an iterator over indexed descriptor key and instance pairs.
        :rtype: collections.Iterator[(collections.Hashable,
                                      smqtk.representation.DescriptorElement)]
        """


def get_descriptor_index_impls(reload_modules=False):
    """
    Discover and return discovered ``DescriptorIndex`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable
          ``DESCRIPTOR_INDEX_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``DESCRIPTOR_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``DescriptorIndex``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = osp.abspath(osp.dirname(__file__))
    env_var = 'DESCRIPTOR_INDEX_PATH'
    helper_var = 'DESCRIPTOR_INDEX_CLASS'
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var,
                              DescriptorIndex, reload_modules=reload_modules)
