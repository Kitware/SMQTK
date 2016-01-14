import cPickle
import os.path as osp

from smqtk.representation import DescriptorIndex
from smqtk.utils import SimpleTimer


__author__ = 'paul.tunison@kitware.com'


class DescriptorMemoryIndex (DescriptorIndex):

    @classmethod
    def is_usable(cls):
        # no dependencies
        return True

    def __init__(self, file_cache=None):
        """
        Initialize a new in-memory descriptor index, or reload one from a
        cache.

        :param file_cache: Optional path to a file path, loading an existing
            index if the file already exists. Either way, providing a path to
            this enabled file caching when descriptors are added to this index.
        :type file_cache: None | str

        """
        super(DescriptorMemoryIndex, self).__init__()

        # Mapping of descriptor UUID to the DescriptorElement instance.
        #: :type: dict[collections.Hashable, smqtk.representation.DescriptorElement]
        self._table = {}
        # Record of optional file cache we're using
        self._file_cache = file_cache

        if file_cache and osp.isfile(file_cache):
            self._log.debug("Loading cached descriptor index table from file: "
                            "%s", file_cache)

            with open(file_cache) as f:
                #: :type: dict[collections.Hashable, smqtk.representation.DescriptorElement]
                self._table = cPickle.load(f)

    def _cache_table(self):
        if self._file_cache:
            with SimpleTimer("Caching descriptor table", self._log.debug):
                with open(self._file_cache, 'wb') as f:
                    cPickle.dump(self._table, f)

    def get_config(self):
        return {
            'file_cache': self._file_cache,
        }

    def count(self):
        return len(self._table)

    def clear(self):
        """
        Clear this descriptor index's entries.
        """
        self._table = {}
        self._cache_table()

    def has_descriptor(self, uuid):
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for
        :type uuid: collections.Hashable

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.
        :rtype: bool

        """
        return uuid in self._table

    def add_descriptor(self, descriptor, no_cache=False):
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple
        copies of the descriptor in the index.

        :param descriptor: Descriptor to index.
        :type descriptor: smqtk.representation.DescriptorElement

        :param no_cache: Do not cache the internal table if a file cache was
            provided. This would be used if adding many descriptors at a time,
            preventing a file write for every individual descriptor added.
        :type no_cache: bool

        """
        self._table[descriptor.uuid()] = descriptor
        if not no_cache:
            self._cache_table()

    def add_many_descriptors(self, descriptors):
        """
        Add multiple descriptors at one time.

        :param descriptors: Iterable of descriptor instances to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        for d in descriptors:
            # using no-cache so we don't trigger multiple file writes
            self.add_descriptor(d, no_cache=True)
        self._cache_table()

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
        return self._table[uuid]

    def get_many_descriptors(self, *uuids):
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated to given uuid values.
        :rtype: collections.Iterable[smqtk.representation.DescriptorElement]

        """
        for uid in uuids:
            yield self._table[uid]

    def remove_descriptor(self, uuid, no_cache=False):
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        :param no_cache: Do not cache the internal table if a file cache was
            provided. This would be used if adding many descriptors at a time,
            preventing a file write for every individual descriptor added.
        :type no_cache: bool

        """
        del self._table[uuid]
        if not no_cache:
            self._cache_table()

    def remove_many_descriptors(self, **uuids):
        """
        Remove descriptors associated to given descriptor UUIDs from this index.

        :param uuids: Iterable of descriptor UUIDs to remove.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """
        for uid in uuids:
            # using no-cache so we don't trigger multiple file writes
            self.remove_descriptor(uid, no_cache=True)
        self._cache_table()

    def iterkeys(self):
        return self._table.iterkeys()

    def iterdescriptors(self):
        return self._table.itervalues()

    def iteritems(self):
        return self._table.iteritems()


DESCRIPTOR_INDEX_CLASS = DescriptorMemoryIndex
