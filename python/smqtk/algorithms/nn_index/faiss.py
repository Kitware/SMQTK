from six.moves import cPickle
import multiprocessing
import tempfile
import os

import numpy as np

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.representation.data_element import from_uri
from smqtk.representation.descriptor_element import elements_to_matrix

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

    def __init__(self, index_uri=None, parameters_uri=None, descriptor_cache_uri=None,
                 exhaustive=False, index_type="IVF100,Flat", nprob=3):
        """
        Initialize MRPT index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.



        :param index_type: index type used for index_factory. (NOTE, we need to give the
            feature dimension in order to use the index factory.)
        :type index_type: str (default: IVF)

        """
        super(FaissNearestNeighborsIndex, self).__init__()

        self._index_uri = index_uri
        self._index_param_uri = parameters_uri
        self._descr_cache_uri = descriptor_cache_uri

        # Elements will be None if input URI is None
        self._index_elem = \
            self._index_uri and from_uri(self._index_uri)
        self._index_param_elem = \
            self._index_param_uri and from_uri(self._index_param_uri)
        self._descr_cache_elem = \
            self._descr_cache_uri and from_uri(self._descr_cache_uri)

        self._exhaustive = exhaustive
        self._index_type = index_type
        self._nprob = nprob

        # In-order cache of descriptors we're indexing over.
        # - flann.nn_index will spit out indices to list
        #: :type: list[smqtk.representation.DescriptorElement] | None
        self._descr_cache = None

        # feature's dimension
        self._feature_dim = None

        self._faiss_index = None

        self._pid = None

    def get_config(self):
        return {
            "index_uri": self._index_uri,
            "parameters_uri": self._index_param_uri,
            "descriptor_cache_uri": self._descr_cache_uri,
            "exhaustive": self.exhaustive,
            "index_type": self.index_type,
            "nprob": self._nprob,
        }

    def _has_model_data(self):
        """
        check if configured model files are configured and not empty
        """
        return (self._index_elem and not self._index_elem.is_empty() and
                self._index_param_elem and not self._index_param_elem.is_empty() and
                self._descr_cache_elem and not self._descr_cache_elem.is_empty())

    def _load_flann_model(self):
        if not self._descr_cache and not self._descr_cache_elem.is_empty():
            # Load descriptor cache
            # - is copied on fork, so only need to load here.
            self._log.debug("Loading cached descriptors")
            self._descr_cache = cPickle.loads(self._descr_cache_elem.get_bytes())

        # Load the binary index
        if self._index_elem and not self._index_elem.is_empty():
            tmp_fp = self._index_elem.write_temp()
            self._faiss_index = faiss.read_index(tmp_fp)
            self._index_elem.clean_temp()
            del tmp_fp

        # Set current PID to the current
        self._pid = multiprocessing.current_process().pid

    def _restore_index(self):
        """
        If we think we're suppose to have an index, check the recorded PID with
        the current PID, reloading the index from cache if they differ.

        If there is a loaded index and we're on the same process that created it
        this does nothing.
        """
        if bool(self._flann) \
                and self._has_model_data() \
                and self._pid != multiprocessing.current_process().pid:
            self._load_flann_model()

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
        self._log.info("Building new FAISS index")

        self._log.debug("Storing descriptors")
        self._descr_cache = list(descriptors)
        if not self._descr_cache:
            raise ValueError("No data provided in given iterable.")

        # Cache descriptors if we have an element
        if self._descr_cache_elem and self._descr_cache_elem.writable():
            self._log.debug("Caching descriptors: %s", self._descr_cache_elem)
            self._descr_cache_elem.set_bytes(
                cPickle.dumps(self._descr_cache, -1)
            )

        n = len(self._descr_cache)
        self._feature_dim = self._descr_cache[0].vector().size

        data = np.empty((n, self._feature_dim), dtype=np.float32)
        elements_to_matrix(self._descr_cache, mat=data, report_interval=1.0)

        if self._exhaustive:
            self._faiss_index = faiss.IndexFlatL2(self._feature_dim)
        else:
            self._faiss_index = faiss.index_factory(self._feature_dim, self._index_type)
            self._faiss_index.train(data)

        if self._index_elem and self._index_elem.writable():
            self._log.debug("Caching index: %s", self._index_elem)
            # FAISS wants to write to a file, so make a temp file, then read it
            # in, putting bytes into element.
            fd, fp = tempfile.mkstemp()
            try:
                faiss.write_index(self._faiss_index, fp)
                self._index_elem.set_bytes(os.read(fd, os.path.getsize(fp)))
            finally:
                os.close(fd)
                os.remove(fp)

        self._pid = multiprocessing.current_process().pid

    def nn(self, d, n=1):
        if self._exhaustive and not self._faiss_index.is_trained:
            raise RuntimeError('The Faiss index is not trained!')

        self._restore_index()
        super(FaissNearestNeighborsIndex, self).nn(d, n)

        # TODO: d can be multiple query descrptors. Currently, we only consider single query.
        query_descr = d.vector()
        query = np.empty((1, self._feature_dim), dtype=np.float32)
        elements_to_matrix([query_descr], mat=query, report_interval=1.0)

        dists, ids = self._faiss_index.search(query, n)
        dists, ids = dists.squeeze(), ids.squeeze()

        return [self._descr_cache[i] for i in ids], dists

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self._descr_cache) if self._descr_cache else 0


NN_INDEX_CLASS = FaissNearestNeighborsIndex
