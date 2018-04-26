from six.moves import cPickle as pickle, zip
from six import next

import numpy as np

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import plugin

__author__ = 'bo.dong@kitware.com'

try:
    import faiss
except ImportError:
    faiss = None

class FaissNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest-neighbor computation using the FAISS library.
    webpage: https://github.com/facebookresearch/faiss
    """

    @classmethod
    def is_usable(cls):
        # if underlying library is not found, the import above will error
        return faiss is not None

    def __init__(self, read_only=False, exhaustive=False, use_multiprocessing=True,
                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                 index_type="IVF"):
        """
        Initialize MRPT index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param read_only: If True, `build_index` will error if there is an
            existing index. False by default.
        :type read_only: bool

        :param num_trees: The number of trees that will be generated for the
            data structure
        :type num_trees: int

        :param depth: The depth of the trees
        :type depth: int

        :param pickle_protocol: The protocol version to be used by the pickle
            module to serialize class information
        :type pickle_protocol: int

        :param use_multiprocessing: Whether or not to use discrete processes
            as the parallelization agent vs python threads.
        :type use_multiprocessing: bool

        :param index_type: index type used for index_factory. (NOTE, we need to give the
            feature dimension in order to use the index factory.)
        :type index_type: str (default: IVF)

        """
        super(FaissNearestNeighborsIndex, self).__init__()

        self.read_only = read_only
        self.use_multiprocessing = use_multiprocessing
        self.pickle_protocol = pickle_protocol
        self.exhaustive = exhaustive
        self.index_type = index_type


    def get_config(self):
        return {
            "descriptor_set": plugin.to_plugin_config(self._descriptor_set),
            "exhaustive": self.exhaustive,
            "read_only": self.read_only,
            "pickle_protocol": self.pickle_protocol,
            "use_multiprocessing": self.use_multiprocessing,
            "index_type": self.index_type
        }

    def build_index(self, descriptors):
        """
        Build the index over the descriptor data elements.

        Subsequent calls to this method should rebuild the index, not add to
        it, or raise an exception to as to protect the current index.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        if self.read_only:
            raise ReadOnlyError("Cannot modify container attributes due to "
                                "being in read-only mode.")

        super(FaissNearestNeighborsIndex, self).build_index(descriptors)

        self._log.info("Building new FAISS index")

        self._log.debug("Clearing and adding new descriptor elements")
        self._descriptor_set.clear()
        self._descriptor_set.add_many_descriptors(descriptors)

        self._log.debug('Building FAISS index')
        self._build_faiss_model()

    def _build_faiss_model(self):
        sample = next(self._descriptor_set.iterdescriptors())
        sample_v = sample.vector()
        n, d = self.count(), sample_v.size

        data = np.empty((n, d), dtype=np.float32)
        elements_to_matrix(
            self._descriptor_set, mat=data,
            use_multiprocessing=self.use_multiprocessing,
            report_interval=1.0,
        )
        self._uuids = np.array(list(self._descriptor_set.keys()))
        self.faiss_flat = faiss.IndexFlatL2(d)

        if self.exhaustive:
            self._faiss_index = faiss.IndexIDMap(self.faiss_flat)
        else:
            nlist = 10000
            self._faiss_index = faiss.IndexIVFFlat(
                self.faiss_flat, d, nlist, faiss.METRIC_L2)
            self._faiss_index.train(data)
            self._faiss_index.nprobe = 5000

        self._log.info("data shape, type: %s, %s", data.shape, data.dtype)
        self._log.info("uuid shape, type: %s, %s",
                       self._uuids.shape, self._uuids.dtype)
        self._faiss_index.add_with_ids(data, self._uuids)

        self._log.info("FAISS index has been constructed with %d vectors",
                       self._faiss_index.ntotal)

    def nn(self, d, n=1):
        super(FaissNearestNeighborsIndex, self).nn(d, n)

        q = d.vector().reshape(1, -1).astype(np.float32)

        self._log.debug("Received query for %d nearest neighbors", n)

        dists, ids = self._faiss_index.search(q, n)
        dists, ids = np.sqrt(dists).squeeze(), ids.squeeze()
        uuids = ids

        descriptors = tuple(self._descriptor_set.get_many_descriptors(uuids))
        d_vectors = elements_to_matrix(descriptors)
        d_dists = np.sqrt(((d_vectors - q)**2).sum(axis=1))

        order = dists.argsort()
        uuids, dists = list(zip(*((uuids[oidx], d_dists[oidx])
                                  for oidx in order)))

        d_dists = d_dists[order]
        self._log.debug("Min and max FAISS distances: %g, %g",
                        min(dists), max(dists))
        self._log.debug("Min and max descriptor distances: %g, %g",
                        min(d_dists), max(d_dists))

        self._log.debug("Returning query result of size %g", len(uuids))

        return (descriptors, tuple(dists))

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self._descriptor_set)

NN_INDEX_CLASS = FaissNearestNeighborsIndex
