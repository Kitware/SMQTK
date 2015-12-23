import abc
import os.path as osp

from smqtk.representation import SmqtkRepresentation
from smqtk.utils import plugin


__author__ = "paul.tunison@kitware.com"


class CodeIndex (SmqtkRepresentation, plugin.Pluggable):
    """
    Abstract interface for bit-code to DescriptorElement relationship storage.

    Implementations should be picklable for serialization.

    """

    def __len__(self):
        return self.count()

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
    def iter_codes(self):
        """
        Iterate over code contained in this index in an arbitrary order.

        :return: Generator that yields integer code keys
        :rtype: collections.Iterator[int|long]

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
        :type descriptor: smqtk.representation.DescriptorElement

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
            collections.Iterable[(int, smqtk.representation.DescriptorElement)]

        """

    @abc.abstractmethod
    def get_descriptors(self, code_or_codes):
        """
        Get iterable of descriptors associated to this code or iterable of
        codes. This may return an empty iterable.

        :param code_or_codes: An integer or iterable of integer bit-codes.
        :type code_or_codes: collections.Iterable[int] | int

        :return: Iterable of descriptors
        :rtype: collections.Iterable[smqtk.representation.DescriptorElement]

        """

    def __getitem__(self, code):
        return self.get_descriptors(code)


def get_code_index_impls(reload_modules=False):
    """
    Discover and return discovered ``CodeIndex`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable
          ``CODE_INDEX_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``CODE_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``CodeIndex``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = osp.abspath(osp.dirname(__file__))
    env_var = 'CODE_INDEX_PATH'
    helper_var = 'CODE_INDEX_CLASS'
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var,
                              CodeIndex, reload_modules=reload_modules)
