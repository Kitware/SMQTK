import abc
import logging

from smqtk.utils.configurable_interface import Configurable


__author__ = 'paul.tunison@kitware.com'


class DescriptorIndex (Configurable):
    """
    Index of descriptors, query-able by descriptor UUID.

    Note that these indexes do not use the descriptor type strings. Thus, if
    an set of descriptors has multiple elements with the same UUID, but
    different type strings, they will bash each other in these indexes. In such
    a case, it is advisable to use multiple indices.

    """

    def __len__(self):
        return self.count()

    def __getitem__(self, uuid):
        return self.get_descriptor(uuid)

    def __delitem__(self, uuid):
        self.remove_descriptor(uuid)

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    @classmethod
    @abc.abstractmethod
    def is_usable(cls):
        """
        Return boolean that describes whether this implementation is available
        for use. If this is false, then it will not be returned as an available
        plugin implementation.

        :return: If this implementation is usable or not.
        :rtype: bool

        """

    @abc.abstractmethod
    def count(self):
        """
        :return: Number of descriptor elements stored in this index. This is not
            necessarily the number of codes stored in the index.
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
    def get_many_descriptors(self, *uuids):
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated to given type-uuid pairs.
        :rtype: collections.Iterable[smqtk.representation.DescriptorElement]

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
    def remove_many_descriptors(self, **uuids):
        """
        Remove descriptors associated to given descriptor UUIDs from this index.

        :param type_uuid_pairs: Iterable of descriptor UUIDs to remove.
        :type type_uuid_pairs: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """

    @abc.abstractmethod
    def iterkeys(self):
        """
        Return an iterator over indexed descriptor keys, which are their
        descriptor type string and UUID: (type_str, uuid)
        """

    @abc.abstractmethod
    def iterdescriptors(self):
        """
        Return an iterator over indexed descriptor element instances.
        """

    @abc.abstractmethod
    def iteritems(self):
        """
        Return an iterator over indexed descriptor key and instance pairs.
        """


def get_descriptor_index_impls(reload_modules=False):
    """
    Discover and return DescriptorIndex implementation classes found in the
    plugin directory.
    Keys in the returned map are the names of the discovered implementations and
    the paired values are the actual class type objects.

    We look for modules (directories or files) that start with and alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module, we first look for a helper variable by the name
    ``DESCRIPTOR_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same na e and casing as the module's
    name. If neither are found, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class objects of type ``DescriptorIndex`` whose
        keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    import os.path as osp
    from smqtk.utils.plugin import get_plugins

    this_dir = osp.abspath(osp.dirname(__file__))
    helper_var = 'DESCRIPTOR_INDEX_CLASS'
    fltr = lambda cls: cls.is_usable()
    return get_plugins(__name__, this_dir, helper_var, DescriptorIndex, fltr,
                       reload_modules)
