__author__ = 'purg'

import abc


class CodeIndex (object):
    """
    Abstract base class for LSH small-code index storage
    """
    __metaclass__ = abc.ABCMeta

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
        return

    @abc.abstractmethod
    def get_descriptors(self, code):
        """
        Get iterable of descriptors associated to this code. This may be empty.

        Runtime: O(1)

        :param code: Integer code bits
        :type code: int

        :return: Iterable of descriptors
        :rtype: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """
        return

    def __getitem__(self, code):
        return self.get_descriptors(code)

    @abc.abstractmethod
    def add_descriptor(self, code, descriptor):
        """
        Add a descriptor to this index given a matching small-code

        :param code: bit-hash of the given descriptor in integer form
        :type code: int

        :param descriptor: Descriptor to index
        :type descriptor: smqtk.data_rep.DescriptorElement

        """
        return

    def __setitem__(self, code, descriptor):
        self.add_descriptor(code, descriptor)


def get_index_types():
    """
    Discover and return small-code index implementation classes found in the
    plugin directory.
    Keys in the returned map are the names of the discovered implementations and
    the paired values are the actual class type objects.

    We look for modules (directories or files) that start with and alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module, we first look for a helper variable by the name
    ``CODE_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same na e and casing as the module's
    name. If neither are found, the module is skipped.

    :return: Map of discovered class objects of type ``CodeIndex`` whose
        keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    import os.path as osp
    from smqtk.utils.plugin import get_plugins

    this_dir = osp.abspath(osp.dirname(__file__))
    helper_var = 'CODE_INDEX_CLASS'
    fltr = lambda cls: cls.is_usable()
    return get_plugins(__name__, this_dir, helper_var, CodeIndex, fltr)
