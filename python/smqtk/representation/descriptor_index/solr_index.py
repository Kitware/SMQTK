import cPickle
import time

from smqtk.representation.descriptor_index import DescriptorIndex

# Try to import required module
try:
    import solr
except ImportError:
    solr = None


__author__ = "paul.tunison@kitware.com"


class SolrDescriptorIndex (DescriptorIndex):
    """
    Descriptor index that uses a Solr instance as a backend storage medium.

    Fields where components are stored within a document are specified at
    construction time. We optionally set the ``id`` field to a string UUID.
    ``id`` is set because it is a common, required field for unique
    identification of documents.

    Descriptor UUIDs should maintain their uniqueness when converted to a
    string, otherwise this backend will not work well when querying.

    """

    @classmethod
    def is_usable(cls):
        return solr is not None

    def __init__(self, solr_conn_addr, index_uuid,
                 index_uuid_field, d_uid_field, descriptor_field,
                 timestamp_field, solr_params=None,
                 commit_on_add=True, max_boolean_clauses=1024,
                 pickle_protocol=-1):
        """
        Construct a descriptor index pointing to a Solr instance.

        :param solr_conn_addr: HTTP(S) address for the Solr index to use
        :type solr_conn_addr: str

        :param index_uuid: Unique ID for the descriptor index to use within the
            configured Solr index.
        :type index_uuid: str

        :param index_uuid_field: Solr index field to store/locate index UUID
            value.
        :type index_uuid_field: str

        :param d_uid_field: Solr index field to store/locate descriptor UUID
            values
        :type d_uid_field: str

        :param descriptor_field: Solr index field to store the code-associated
            descriptor object.
        :type descriptor_field: str

        :param timestamp_field: Solr index field to store floating-point UNIX
            timestamps.
        :type timestamp_field: str

        :param solr_params: Dictionary of additional keyword parameters to set
            in the ``solr.Solr`` instance used. See the ``pysolr``
            documentation for available parameters and values.
        :type solr_params: dict[str, object]

        :param commit_on_add: Immediately commit changes when one or many
            descriptor are added.
        :type commit_on_add: bool

        :param max_boolean_clauses: Solr instance's configured
            maxBooleanClauses configuration property (found in solrconfig.xml
            file). This is needed so we can correctly chunk up batch queries
            without breaking the server. This may also be less than the Solr
            instance's set value.
        :type max_boolean_clauses: int

        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :type pickle_protocol: int

        """
        super(SolrDescriptorIndex, self).__init__()

        self.index_uuid = index_uuid

        self.index_uuid_field = index_uuid_field
        self.d_uid_field = d_uid_field
        self.descriptor_field = descriptor_field
        self.timestamp_field = timestamp_field

        self.commit_on_add = commit_on_add
        self.max_boolean_clauses = int(max_boolean_clauses)
        assert self.max_boolean_clauses >= 2, "Need more clauses"

        self.pickle_protocol = pickle_protocol

        self.solr_params = solr_params
        self.solr = solr.Solr(solr_conn_addr, **solr_params)

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, state):
        state['solr'] = solr.Solr(state["solr_conn_addr"],
                                  **state['solr_params'])
        del state['solr_conn_addr']
        self.__dict__.update(state)

    def _doc_for_code_descr(self, d):
        """
        Generate standard identifying document base for the given
        descriptor element.
        """
        uuid = d.uuid()
        return {
            'id': '-'.join([self.index_uuid,  uuid]),
            self.index_uuid_field: self.index_uuid,
            self.d_uid_field: uuid,
        }

    def get_config(self):
        return {
            "solr_conn_addr": self.solr.url,
            "index_uuid": self.index_uuid,
            "index_uuid_field": self.index_uuid_field,
            "d_uid_field": self.d_uid_field,
            "descriptor_field": self.descriptor_field,
            "timestamp_field": self.timestamp_field,
            "solr_params": self.solr_params,
            "commit_on_add": self.commit_on_add,
            "max_boolean_clauses": self.max_boolean_clauses,
            "pickle_protocol": self.pickle_protocol,
        }

    def count(self):
        """
        :return: Number of descriptor elements stored in this index.
        :rtype: int
        """
        return int(self.solr.
                   select("%s:%s AND %s:*"
                          % (self.index_uuid_field, self.index_uuid,
                             self.descriptor_field))
                   .numFound)

    def clear(self):
        """
        Clear this descriptor index's entries.
        """
        self.solr.delete_query("%s:%s"
                               % (self.index_uuid_field, self.index_uuid))
        self.solr.commit()

    def has_descriptor(self, uuid):
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for
        :type uuid: collections.Hashable

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.
        :rtype: bool

        """
        # Try to select the descriptor
        # TODO: Probably a better way of doing this that's more efficient.
        return bool(
            self.solr.select("%s:%s AND %s:%s"
                             % (self.index_uuid_field, self.index_uuid,
                                self.d_uid_field, uuid)).numFound
        )

    def add_descriptor(self, descriptor):
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptor: Descriptor to index.
        :type descriptor: smqtk.representation.DescriptorElement

        """
        doc = self._doc_for_code_descr(descriptor)
        doc[self.descriptor_field] = cPickle.dumps(descriptor,
                                                   self.pickle_protocol)
        doc[self.timestamp_field] = time.time()
        self.solr.add(doc, commit=self.commit_on_add)

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
        documents = []
        for d in descriptors:
            doc = self._doc_for_code_descr(d)
            doc[self.descriptor_field] = cPickle.dumps(d, self.pickle_protocol)
            doc[self.timestamp_field] = time.time()
            documents.append(doc)
        self.solr.add_many(documents)
        if self.commit_on_add:
            self.solr.commit()

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
        return tuple(self.get_many_descriptors(uuid))[0]

    def get_many_descriptors(self, *uuids):
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated 1-to-1 to given uuid values.
        :rtype: __generator[smqtk.representation.DescriptorElement]

        """
        # Chunk up query based on max clauses available to us

        def batch_query(batch):
            """
            :type batch: list[collections.Hashable]
            """
            query = ' OR '.join([self.d_uid_field + (':%s' % uid)
                                 for uid in batch])
            r = self.solr.select("%s:%s AND (%s)"
                                 % (self.index_uuid_field, self.index_uuid,
                                    query))
            # result batches come in chunks of 10
            for doc in r.results:
                yield cPickle.loads(doc[self.descriptor_field])
            for j in xrange(r.numFound // 10):
                r = r.next_batch()
                for doc in r.results:
                    yield cPickle.loads(doc[self.descriptor_field])

        batch = []
        for uid in uuids:
            batch.append(uid)

            # Will end up using max_clauses-1 OR statements, and one AND
            if len(batch) == self.max_boolean_clauses:
                for d in batch_query(batch):
                    yield d
                batch = []

        # tail batch
        if batch:
            assert len(batch) < self.max_boolean_clauses
            for d in batch_query(batch):
                yield d

    def remove_descriptor(self, uuid):
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        """
        self.remove_many_descriptors(uuid)

    def remove_many_descriptors(self, uuids):
        """
        Remove descriptors associated to given descriptor UUIDs from this index.

        :param uuids: Iterable of descriptor UUIDs to remove.
        :type uuids: tuple[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """
        # Chunk up operation based on max clauses available to us

        def batch_op(batch):
            """
            :type batch: list[collections.Hashable]
            """
            uuid_query = ' OR '.join([self.d_uid_field + (':%s' % str(uid))
                                      for uid in batch])
            self.solr.delete("%s:%s AND (%s)"
                             % (self.index_uuid_field, self.index_uuid,
                                uuid_query))

        batch = []
        for uid in uuids:
            batch.append(uid)

            # Will end up using max_clauses-1 OR statements, and one AND
            if len(batch) == self.max_boolean_clauses:
                batch_op(batch)
            batch = []

        # tail batch
        if batch:
            batch_op(batch)

    def iterkeys(self):
        """
        Return an iterator over indexed descriptor keys, which are their UUIDs.
        """
        r = self.solr.select('%s:%s %s:*'
                             % (self.index_uuid_field, self.index_uuid,
                                self.d_uid_field))
        for doc in r.results:
            yield doc[self.d_uid_field]
        for _ in xrange(r.numFound // 10):
            r = r.next_batch()
            for doc in r.results:
                yield doc[self.d_uid_field]

    def iterdescriptors(self):
        """
        Return an iterator over indexed descriptor element instances.
        """
        r = self.solr.select('%s:%s %s:*'
                             % (self.index_uuid_field, self.index_uuid,
                                self.descriptor_field))
        for doc in r.results:
            yield cPickle.loads(doc[self.descriptor_field])
        for _ in xrange(r.numFound // 10):
            r = r.next_batch()
            for doc in r.results:
                yield cPickle.loads(doc[self.descriptor_field])

    def iteritems(self):
        """
        Return an iterator over indexed descriptor key and instance pairs.
        """
        r = self.solr.select('%s:%s %s:* %s:*'
                             % (self.index_uuid_field, self.index_uuid,
                                self.d_uid_field, self.descriptor_field))
        for doc in r.results:
            d = cPickle.loads(doc[self.descriptor_field])
            yield d.uuid(), d
        for _ in xrange(r.numFound // 10):
            r = r.next_batch()
            for doc in r.results:
                d = cPickle.loads(doc[self.descriptor_field])
                yield d.uuid(), d
