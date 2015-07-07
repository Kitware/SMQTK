__author__ = 'purg'

from . import CodeIndex


class MemoryCodeIndex (CodeIndex):
    """
    Local RAM memory based index. This implementation cannot save state.
    """

    @classmethod
    def is_usable(cls):
        """
        No outside dependencies.
        :rtype: bool
        """
        return True

    def __init__(self):
        self._num_descr = 0
        # Mapping of code to a dictionary mapping descrUUID->Descriptor
        self._table = {}

    def count(self):
        """
        :return: Number of descriptor elements stored in this index. This is not
            necessarily the number of codes stored in the index.
        :rtype: int
        """
        return self._num_descr

    def add_descriptor(self, code, descriptor):
        """
        Add a descriptor to this index given a matching small-code

        :param code: bit-hash of the given descriptor in integer form
        :type code: int

        :param descriptor: Descriptor to index
        :type descriptor: smqtk.data_rep.DescriptorElement

        """
        self._table.setdefault(code, {})[descriptor.uuid()] = descriptor
        self._num_descr += 1

    def get_descriptors(self, code):
        """
        Get iterable of descriptors associated to this code. This may be empty.

        Runtime: O(1)

        :param code: Integer code bits
        :type code: int

        :return: Iterable of descriptors
        :rtype: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """
        return self._table.get(code, {}).values()


CODE_INDEX_CLASS = MemoryCodeIndex
