__author__ = 'purg'

import logging
import solr
from smqtk.data_rep import DescriptorElement
import time
import uuid


class SolrDescriptorElement (DescriptorElement):

    def __init__(self, type_str, uid, solr_conn_addr, type_field, uuid_field,
                 vector_field,
                 timeout=10, persistent_connection=False, commit_on_set=True):
        """
        Initialize a new Solr-stored descriptor element.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uid: Unique ID reference of the descriptor.
        :type uid: collections.Hashable

        :param solr_conn_addr: HTTP(S) address for the Solr index to use
        :type solr_conn_addr: str

        :param type_field: Solr index field to store descriptor type string
            value.
        :type type_field: str

        :param uuid_field: Solr index field to store descriptor UUID string
            value in.
        :type uuid_field: str

        :param vector_field: Solr index field to store the descriptor vector of
            floats in.
        :type vector_field: str

        :param timeout: Whether or not the Solr connection should
            be persistent or not.
        :type timeout: int

        :param persistent_connection: Maintain a connection between Solr index
            interactions.
        :type persistent_connection: bool

        :param commit_on_set: Immediately commit changes when a vector is set.
        :type commit_on_set: bool

        """
        super(SolrDescriptorElement, self).__init__(type_str, uid)

        self.type_field = type_field
        self.uuid_field = uuid_field
        self.vector_field = vector_field
        self.commit_on_set = commit_on_set
        self.solr = solr.Solr(solr_conn_addr,
                              persistent=persistent_connection,
                              timeout=timeout, debug=self._is_debug())

    def _is_debug(self):
        is_debug = False
        if self._log.getEffectiveLevel() <= logging.DEBUG:
            is_debug = True
        return is_debug

    def __getstate__(self):
        return {
            "type_label": self._type_label,
            "uuid": self._uuid,
            "vector_field": self.vector_field,
            "type_field": self.type_field,
            "uuid_field": self.uuid_field,
            "commit_on_set": self.commit_on_set,
            "solr_url": self.solr.url,
            "solr_persistent": self.solr.persistent,
            "solr_timeout": self.solr.timeout,
        }

    def __setstate__(self, state):
        self._type_label = state['type_label']
        self._uuid = state['uuid']
        self.vector_field = state['vector_field']
        self.type_field = state['type_field']
        self.uuid_field = state['uuid_field']
        self.commit_on_set = state['commit_on_set']
        self.solr = solr.Solr(state['solr_url'],
                              persistent=state['solr_persistent'],
                              timeout=state['solr_timeout'],
                              debug=self._is_debug())

    def __repr__(self):
        return super(SolrDescriptorElement, self).__repr__() + \
            '[url: %s, vector_field: %s, timeout: %d, ' \
            'persistent: %s]' \
            % (self.solr.url, self.vector_field,
               self.solr.timeout, self.solr.persistent)

    def _query_str(self):
        """
        :return: Standard solr query string for this element's unique
            combination of keys.
        :rtype: str
        """
        return "%s:%s AND %s:%s" % (self.type_field, self.type(),
                                    self.uuid_field, str(self.uuid()))

    def _base_doc(self):
        return {
            'id': uuid.uuid1(clock_seq=int(time.time() * 1000000)),
            self.type_field: self.type(),
            self.uuid_field: str(self.uuid()),
        }

    def _get_existing_doc(self):
        """
        :return: An existing document dict. If there isn't one for our type/uuid
            we return None.
        :rtype: None | dict
        """
        r = self.solr.select(self._query_str())
        if r.numFound == 1:
            return r.results[0]
        else:
            return None

    def has_vector(self):
        return bool(self._get_existing_doc())

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray

        """
        doc = self._get_existing_doc()
        if doc is None:
            doc = self._base_doc()
        doc[self.vector_field] = new_vec.tolist()
        self.solr.add(doc, commit=self.commit_on_set)

    def vector(self):
        doc = self._get_existing_doc()
        if doc is None:
            return None
        return doc[self.vector_field]


DESCRIPTOR_ELEMENT_CLASS = SolrDescriptorElement
