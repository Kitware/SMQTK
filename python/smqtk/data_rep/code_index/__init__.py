
import abc
import logging

from smqtk.utils.configurable_interface import Configurable


__author__ = 'purg'


class CodeIndex (Configurable):
    """
    Abstract base class for LSH small-code index storage

    Implementations should be picklable for serialization.

    """
    __metaclass__ = abc.ABCMeta

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __len__(self):
        return self.count()

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
        Clear this code index's entries.
        """

    @abc.abstractmethod
    def codes(self):
        """
        :return: Set of code integers currently used in this code index.
        :rtype: set[int]
        """

    @abc.abstractmethod
    def add_descriptor(self, code, descriptor):
        """
        Add a descriptor to this index given a matching small-code.

        Adding the same descriptor multiple times under the same code should not
        add multiple copies of the descriptor in the index.

        :param code: bit-hash of the given descriptor in integer form
        :type code: int

        :param descriptor: Descriptor to index
        :type descriptor: smqtk.data_rep.DescriptorElement

        """

    def __setitem__(self, code, descriptor):
        self.add_descriptor(code, descriptor)

    @abc.abstractmethod
    def add_many_descriptors(self, code_descriptor_pairs):
        """
        Add multiple code/descriptor pairs.

        :param code_descriptor_pairs: Iterable of integer code and paired
            descriptor tuples to add to this index.
        :type code_descriptor_pairs:
            collections.Iterable[(int, smqtk.data_rep.DescriptorElement)]

        """
        return

    @abc.abstractmethod
    def get_descriptors(self, code_or_codes):
        """
        Get iterable of descriptors associated to this code or iterable of
        codes. This may return an empty iterable.

        Runtime: O(n) where n is the number of codes provided.

        :param code_or_codes: An integer or iterable of integer bit-codes.
        :type code_or_codes: collections.Iterable[int] | int

        :return: Iterable of descriptors
        :rtype: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """

    def __getitem__(self, code):
        return self.get_descriptors(code)


def get_index_types(reload_modules=False):
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

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class objects of type ``CodeIndex`` whose
        keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    import os.path as osp
    from smqtk.utils.plugin import get_plugins

    this_dir = osp.abspath(osp.dirname(__file__))
    helper_var = 'CODE_INDEX_CLASS'
    fltr = lambda cls: cls.is_usable()
    return get_plugins(__name__, this_dir, helper_var, CodeIndex, fltr,
                       reload_modules)
