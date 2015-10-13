import numpy
from smqtk.representation import DescriptorElement
import time


# Try to import required module
try:
    import solr
except ImportError:
    solr = None


__author__ = "paul.tunison@kitware.com"


class SolrDescriptorElement (DescriptorElement):
    """
    Descriptor element that uses a Solr instance as the backend storage medium.

    Fields where data is stored in the Solr documents are specified at
    construction time. We additionally set the ``id`` field to a string UUID.
    ``id`` is set because it is a common, required field for unique
    identification of documents. The value set to the ``id`` field is
    reproducible from this object's key attributes.

    """

    @classmethod
    def is_usable(cls):
        return solr is not None

    def __init__(self, type_str, uuid, solr_conn_addr,
                 type_field, uuid_field, vector_field, timestamp_field,
                 timeout=10, persistent_connection=False, commit_on_set=True):
        """
        Initialize a new Solr-stored descriptor element.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: Unique ID reference of the descriptor.
        :type uuid: collections.Hashable

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

        :param timestamp_field: Solr index field to store floating-point UNIX
            timestamps.
        :type timestamp_field: str

        :param timeout: Whether or not the Solr connection should
            be persistent or not.
        :type timeout: int

        :param persistent_connection: Maintain a connection between Solr index
            interactions.
        :type persistent_connection: bool

        :param commit_on_set: Immediately commit changes when a vector is set.
        :type commit_on_set: bool

        """
        super(SolrDescriptorElement, self).__init__(type_str, uuid)

        self.type_field = type_field
        self.uuid_field = uuid_field
        self.vector_field = vector_field
        self.timestamp_field = timestamp_field
        self.commit_on_set = commit_on_set

        self.solr = solr.Solr(solr_conn_addr,
                              persistent=persistent_connection,
                              timeout=timeout,
                              # debug=True  # This makes things pretty verbose
                              )

    def __getstate__(self):
        return {
            "type_label": self._type_label,
            "uuid": self._uuid,
            "type_field": self.type_field,
            "uuid_field": self.uuid_field,
            "vector_field": self.vector_field,
            "timestamp_field": self.timestamp_field,
            "commit_on_set": self.commit_on_set,
            "solr_url": self.solr.url,
            "solr_persistent": self.solr.persistent,
            "solr_timeout": self.solr.timeout,
        }

    def __setstate__(self, state):
        self._type_label = state['type_label']
        self._uuid = state['uuid']
        self.type_field = state['type_field']
        self.uuid_field = state['uuid_field']
        self.vector_field = state['vector_field']
        self.timestamp_field = state['timestamp_field']
        self.commit_on_set = state['commit_on_set']
        self.solr = solr.Solr(state['solr_url'],
                              persistent=state['solr_persistent'],
                              timeout=state['solr_timeout'],
                              # debug=True  # see above
                              )

    def __repr__(self):
        return super(SolrDescriptorElement, self).__repr__() + \
            '[url: %s, timeout: %d, ' \
            'persistent: %s]' \
            % (self.solr.url, self.solr.timeout, self.solr.persistent)

    def _base_doc(self):
        t = self.type()
        suuid = str(self.uuid())
        return {
            'id': '-'.join([t, suuid]),
            self.type_field: t,
            self.uuid_field: suuid,
        }

    def _get_existing_doc(self):
        """
        :return: An existing document dict. If there isn't one for our type/uuid
            we return None.
        :rtype: None | dict
        """
        b_doc = self._base_doc()
        r = self.solr.select("id:%s AND %s:%s AND %s:%s"
                             % (b_doc['id'],
                                self.type_field, b_doc[self.type_field],
                                self.uuid_field, b_doc[self.uuid_field]))
        if r.numFound == 1:
            return r.results[0]
        else:
            return None

    def get_config(self):
        return {
            "solr_conn_addr": self.solr.url,
            "type_field": self.type_field,
            "uuid_field": self.uuid_field,
            "vector_field": self.vector_field,
            "timestamp_field": self.timestamp_field,
            "timeout": self.solr.timeout,
            "persistent_connection": self.solr.persistent,
            "commit_on_set": self.commit_on_set,
        }

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
        doc = self._base_doc()
        doc[self.vector_field] = new_vec.tolist()
        doc[self.timestamp_field] = time.time()
        self.solr.add(doc, commit=self.commit_on_set)

    def vector(self):
        doc = self._get_existing_doc()
        if doc is None:
            return None
        # Vectors stored as lists in solr doc
        return numpy.array(doc[self.vector_field])


DESCRIPTOR_ELEMENT_CLASS = SolrDescriptorElement
