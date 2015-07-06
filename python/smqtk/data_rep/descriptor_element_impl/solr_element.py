__author__ = 'purg'

import logging
import multiprocessing
import solr
from smqtk.data_rep import DescriptorElement


class SolrDescriptorElement (DescriptorElement):

    def __init__(self, type_str, uuid, solr_conn_addr, vector_field,
                 uuid_header='', timeout=10, persistent_connection=False):
        """
        Initialize a new Solr-stored descriptor element.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: Unique ID reference of the descriptor.
        :type uuid: collections.Hashable

        :param solr_conn_addr: HTTP(S) address for the Solr index to use
        :type solr_conn_addr: str

        :param vector_field: Solr index field to store the descriptor vector of
            floats in.
        :type vector_field: str

        :param uuid_header: Custom string to append to the beginning of the UUID
            when forming the ID of the stored descriptor entry. By default there
            no header string (empty string).
        :type uuid_header: str

        :param timeout: Whether or not the Solr connection should
            be persistent or not.
        :type timeout: int

        :param persistent_connection: Maintain a connection between Solr index
            interactions.
        :type persistent_connection: bool

        """
        super(SolrDescriptorElement, self).__init__(type_str, uuid)

        self._lock = multiprocessing.RLock()

        is_debug = False
        if self._log.getEffectiveLevel() <= logging.DEBUG:
            is_debug = True

        self.uuid_header = uuid_header
        self.vector_field = vector_field
        self.solr = solr.Solr(solr_conn_addr,
                              persistent=persistent_connection,
                              timeout=timeout, debug=is_debug)

    def __repr__(self):
        return super(SolrDescriptorElement, self).__repr__() + \
            '[url: %s, vector_field: %s, uuid_header: %s, timeout: %d, ' \
            'persistent: %s]' \
            % (self.solr.url, self.vector_field, self.uuid_header,
               self.solr.timeout, self.solr.persistent)

    def _make_id(self):
        return self.uuid_header + str(self.uuid())

    def has_vector(self):
        with self._lock:
            d_id = self._make_id()
            #: :type: solr.core.Response
            r = self.solr.select('id:%s' % d_id)
            if r.numFound == 1:
                if self.vector_field in r.results[0]:
                    return True
                else:
                    self._log.warn("Descriptor vector field not found! ('%s'",
                                   self.vector_field)
                    return False
            elif r.numFound > 1:
                self._log.warn("Found more than one entry for constructed id: "
                               "%s" % d_id)
                return False
            else:
                self._log.warn("No descriptor found for given id[%s]" % d_id)
                return False

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray

        """
        with self._lock:
            self.solr.add({
                'id': self._make_id(),
                self.vector_field: new_vec,
            })

    def vector(self):
        with self._lock:
            #: :type: solr.core.Response
            r = self.solr.select('id:%s' % self._make_id())
            assert r.numFound == 1
            return r.results[0][self.vector_field]


DESCRIPTOR_ELEMENT_CLASS = SolrDescriptorElement
