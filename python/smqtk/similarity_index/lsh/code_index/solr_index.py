__author__ = 'purg'

import cPickle
import solr
import time
import uuid

from . import CodeIndex


class SolrCodeIndex (CodeIndex):

    @classmethod
    def is_usable(cls):
        # Only needs solrpy to be installed, which is part of the
        # requirements.pip.txt
        return True

    def __init__(self, solr_conn_addr, index_uuid,
                 idx_uuid_field, code_field, d_uid_field, descriptor_field,
                 timeout=10, persistent_connection=False,
                 commit_on_add=True):
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

        :param timeout: Whether or not the Solr connection should
            be persistent or not.
        :type timeout: int

        :param persistent_connection: Maintain a connection between Solr index
            interactions.
        :type persistent_connection: bool

        :param commit_on_add: Immediately commit changes when a vector is set.
        :type commit_on_add: bool

        """
        self.uuid = index_uuid

        self.idx_uuid_field = idx_uuid_field
        self.code_field = code_field
        self.d_uid_field = d_uid_field
        self.descriptor_field = descriptor_field

        self.commit_on_add = commit_on_add

        self.solr = solr.Solr(solr_conn_addr, persistent=persistent_connection,
                              timeout=timeout,
                              # debug=True
                              )

    # TODO: Pickle state save/load methods

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

    def add_descriptor(self, code, descriptor):
        """
        Add a descriptor to this index given a matching small-code

        :param code: bit-hash of the given descriptor in integer form
        :type code: int

        :param descriptor: Descriptor to index
        :type descriptor: smqtk.data_rep.DescriptorElement

        """
        doc = {
            'id': str(uuid.uuid1(clock_seq=int(time.time() * 1000000))),
            self.idx_uuid_field: self.uuid,
            self.code_field: code,
            self.d_uid_field: str(descriptor.uuid()),
        }

        # Find if there is already an existing document for the given
        # code-descriptor pairing
        r = self.solr.select('%s:%s AND %s:%d AND %s:%s'
                             % (self.idx_uuid_field, self.uuid,
                                self.code_field, code,
                                self.d_uid_field, str(descriptor.uuid())))
        if r.numFound == 1:
            doc['id'] = r.results[0]['id']

        doc[self.descriptor_field] = cPickle.dumps(descriptor)
        self.solr.add(doc, commit=self.commit_on_add)

    def get_descriptors(self, code):
        """
        Get iterable of descriptors associated to this code. This may be empty.

        Runtime: O(1)

        :param code: Integer code bits
        :type code: int

        :return: Iterable of descriptors
        :rtype: collections.Iterable[smqtk.data_rep.DescriptorElement]

        """
        # Get all descriptors for our UUID and the code provided
        r = self.solr.select("%s:%s AND %s:%d"
                             % (self.idx_uuid_field, self.uuid,
                                self.code_field, code))
        pickled_descr = [doc[self.descriptor_field] for doc in r.results]
        return [cPickle.loads(str(pd)) for pd in pickled_descr]
