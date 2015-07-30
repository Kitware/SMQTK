__author__ = 'purg'

import cPickle
import solr
import time

from . import CodeIndex


class SolrCodeIndex (CodeIndex):
    """
    Code index that uses a Solr instance as a backend storage medium.

    Fields where components are stored within a document are specified at
    construction time. We additionally set the ``id`` field to a string UUID.
    ``id`` is set because it is a common, required field for unique
    identification of documents.

    """

    @classmethod
    def is_usable(cls):
        # Only needs solrpy to be installed, which is part of the
        # requirements.pip.txt
        return True

    def __init__(self, solr_conn_addr, index_uuid,
                 idx_uuid_field, code_field, d_uid_field, descriptor_field,
                 timestamp_field,
                 timeout=10, persistent_connection=False,
                 commit_on_add=True, max_boolean_clauses=1024):
        """
        Construct a bit-code index, pointing to a Solr index.

        :param solr_conn_addr: HTTP(S) address for the Solr index to use
        :type solr_conn_addr: str

        :param index_uuid: Unique ID for the index to use within the configured
            Solr index
        :type index_uuid: str

        :param idx_uuid_field: Solr index field to store/locate index UUID
            value.
        :type idx_uuid_field: str

        :param code_field: Solr index field to store the bit-code
        :type code_field: str

        :param d_uid_field: Solr index field to store/locate descriptor UUID
            values
        :type d_uid_field: str

        :param descriptor_field: Solr index field to store the code-associated
            descriptor object.
        :type descriptor_field: str

        :param timestamp_field: Solr index field to store floating-point UNIX
            timestamps.
        :type timestamp_field: str

        :param timeout: Whether or not the Solr connection should
            be persistent or not.
        :type timeout: int

        :param persistent_connection: Maintain a connection between Solr index
            interactions.
        :type persistent_connection: bool

        :param commit_on_add: Immediately commit changes when a vector is set.
        :type commit_on_add: bool

        :param max_boolean_clauses: Solr instance's configured maxBooleanClauses
            configuration property (found in solrconfig.xml file). This is
            needed so we can correctly chunk up batch queries without breaking
            the server.
        :type max_boolean_clauses: int

        """
        self.uuid = index_uuid

        self.idx_uuid_field = idx_uuid_field
        self.code_field = code_field
        self.d_uid_field = d_uid_field
        self.descriptor_field = descriptor_field
        self.timestamp_field = timestamp_field

        self.commit_on_add = commit_on_add
        self.max_boolean_clauses = max_boolean_clauses
        assert self.max_boolean_clauses >= 2, "Need more clauses"

        self.solr = solr.Solr(solr_conn_addr, persistent=persistent_connection,
                              timeout=timeout,
                              # debug=True
                              )

    def __getstate__(self):
        return {
            "uuid": self.uuid,
            "commit_on_add": self.commit_on_add,
            "max_boolean_clauses": self.max_boolean_clauses,
            "field_uuid": self.idx_uuid_field,
            "field_code": self.code_field,
            "field_descr_uuid": self.d_uid_field,
            "field_descr_obj": self.descriptor_field,
            "field_timestamp": self.timestamp_field,
            "solr_url": self.solr.url,
            "solr_timeout": self.solr.timeout,
            "solr_persistent": self.solr.persistent,
        }

    def __setstate__(self, state):
        self.uuid = state['uuid']
        self.commit_on_add = state['commit_on_add']
        self.max_boolean_clauses = state['max_boolean_clauses']
        self.idx_uuid_field = state['field_uuid']
        self.code_field = state['field_code']
        self.d_uid_field = state['field_descr_uuid']
        self.descriptor_field = state['field_descr_obj']
        self.timestamp_field = state['field_timestamp']

        self.solr = solr.Solr(state['solr_url'],
                              persistent=state['solr_persistent'],
                              timeout=state['solr_timeout'])

    def count(self):
        """
        :return: Number of descriptor elements stored in this index. This is not
            necessarily the number of codes stored in the index.
        :rtype: int
        """
        return int(self.solr.
                   select("%s:%s AND %s:* AND %s:*"
                          % (self.idx_uuid_field, self.uuid,
                             self.code_field,
                             self.descriptor_field))
                   .numFound)

    def _doc_for_code_descr(self, code, descr):
        """
        Generate standard identifying document base for the given
        code/descriptor combination.
        """
        scode = str(code)
        sduuid = str(descr.uuid())
        return {
            'id': '-'.join([self.uuid, scode, sduuid]),
            self.idx_uuid_field: self.uuid,
            self.code_field: code,
            self.d_uid_field: sduuid
        }

    def add_descriptor(self, code, descriptor):
        """
        Add a descriptor to this index given a matching small-code

        :param code: bit-hash of the given descriptor in integer form
        :type code: int

        :param descriptor: Descriptor to index
        :type descriptor: smqtk.data_rep.DescriptorElement

        """
        doc = self._doc_for_code_descr(code, descriptor)
        doc[self.descriptor_field] = cPickle.dumps(descriptor)
        doc[self.timestamp_field] = time.time()
        self.solr.add(doc, commit=self.commit_on_add)

    def add_many_descriptors(self, code_descriptor_pairs):
        """
        Add multiple code/descriptor pairs.

        :param code_descriptor_pairs: Iterable of integer code and paired
            descriptor tuples to add to this index.
        :type code_descriptor_pairs:
            collections.Iterable[(int, smqtk.data_rep.DescriptorElement)]

        """
        documents = []
        for c, d in code_descriptor_pairs:
            doc = self._doc_for_code_descr(c, d)
            doc[self.descriptor_field] = cPickle.dumps(d)
            doc[self.timestamp_field] = time.time()
            documents.append(doc)
        self.solr.add_many(documents)
        if self.commit_on_add:
            self.solr.commit()

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
        # Get all descriptors for our UUID and the code provided
        if not hasattr(code_or_codes, '__iter__'):
            code_or_codes = [code_or_codes]
        codes = list(code_or_codes)

        pickled_descr = []

        # Chunk up query based on max clauses available to us
        max_ors = self.max_boolean_clauses - 1
        for i in xrange((len(codes) // max_ors) + 1):
            code_chunk = codes[i*max_ors: (i+1)*max_ors]
            code_query = ' OR '.join([self.code_field + (':%d' % c)
                                      for c in code_chunk])
            r = self.solr.select("%s:%s AND (%s)"
                                 % (self.idx_uuid_field, self.uuid, code_query))
            # result batches come in chunks of 10
            pickled_descr.extend(doc[self.descriptor_field]
                                 for doc in r.results)
            for j in xrange(r.numFound // 10):
                r = r.next_batch()
                pickled_descr.extend(doc[self.descriptor_field]
                                     for doc in r.results)

        return [cPickle.loads(str(pd)) for pd in pickled_descr]
