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

    def codes(self):
        """
        :return: Set of code integers currently used in this code index.
        :rtype: set[int]
        """
        return set(self._table)

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

    def add_many_descriptors(self, code_descriptor_pairs):
        """
        Add multiple code/descriptor pairs.

        :param code_descriptor_pairs: Iterable of integer code and paired
            descriptor tuples to add to this index.
        :type code_descriptor_pairs:
            collections.Iterable[(int, smqtk.data_rep.DescriptorElement)]

        """
        for c, d in code_descriptor_pairs:
            self.add_descriptor(c, d)

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
        if hasattr(code_or_codes, '__iter__'):
            # noinspection PyTypeChecker
            # -> I literally just checked for __iter__
            for c in code_or_codes:
                for v in self._table.get(c, {}).values():
                    yield v
        else:  # assuming int
            for v in self._table.get(code_or_codes, {}).itervalues():
                yield v


CODE_INDEX_CLASS = MemoryCodeIndex
