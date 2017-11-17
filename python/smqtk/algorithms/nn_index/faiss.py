from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

# noinspection PyPep8Naming
from six.moves import range, cPickle as pickle, zip

import logging
import multiprocessing
import os.path as osp

import numpy as np

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation import get_descriptor_index_impls
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import plugin, merge_dict
from smqtk.utils.file_utils import safe_create_dir


# Requires FAISS bindings
try:
    import faiss
except ImportError:
    faiss = None


class FaissNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest-neighbor computation using the FAISS library.

    SUPPORT FOR THIS FUNCTIONALITY IS EXPERIMENTAL AT THIS STAGE. THERE ARE 
    NO TESTS AND THE IMPLEMENTATION DOES NOT COVER ALL OF THE FUNCTIONALITY 
    OF THE FAISS LIBRARY.
    """

    @classmethod
    def is_usable(cls):
        # if underlying library is not found, the import above will error
        return faiss is not None

    def __init__(self, descriptor_set, read_only=False, exhaustive=False,
                 use_multiprocessing=True,
                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                 random_seed=None):
        """
        Initialize MRPT index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param descriptor_set: Index in which DescriptorElements will be
            stored.
        :type descriptor_set: smqtk.representation.DescriptorIndex

        :param read_only: If True, `build_index` will error if there is an
            existing index. False by default.
        :type read_only: bool

        :param num_trees: The number of trees that will be generated for the
            data structure
        :type num_trees: int

        :param depth: The depth of the trees
        :type depth: int

        :param random_seed: Integer to use as the random number generator
            seed.
        :type random_seed: int

        :param pickle_protocol: The protocol version to be used by the pickle
            module to serialize class information
        :type pickle_protocol: int

        :param use_multiprocessing: Whether or not to use discrete processes
            as the parallelization agent vs python threads.
        :type use_multiprocessing: bool

        """
        super(FaissNearestNeighborsIndex, self).__init__()

        self.read_only = read_only
        self.use_multiprocessing = use_multiprocessing
        self.pickle_protocol = pickle_protocol
        self.exhaustive = exhaustive
        self._descriptor_set = descriptor_set

        def normpath(p):
            return (p and osp.abspath(osp.expanduser(p))) or p

        self.random_seed = None
        if random_seed is not None:
            self.random_seed = int(random_seed)

    def get_config(self):
        return {
            "descriptor_set": plugin.to_plugin_config(self._descriptor_set),
            "exhaustive": self.exhaustive,
            "read_only": self.read_only,
            "random_seed": self.random_seed,
            "pickle_protocol": self.pickle_protocol,
            "use_multiprocessing": self.use_multiprocessing,
        }

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: LSHNearestNeighborIndex

        """
        if merge_default:
            cfg = cls.get_default_config()
            merge_dict(cfg, config_dict)
        else:
            cfg = config_dict

        cfg['_descriptor_set'] = \
            plugin.from_plugin_config(cfg['_descriptor_set'],
                                      get_descriptor_index_impls())

        return super(FaissNearestNeighborsIndex, cls).from_config(cfg, False)

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
        sample = self._descriptor_set.iterdescriptors().next()
        sample_v = sample.vector()
        n, d = self.count(), sample_v.size

        data = np.empty((n, d), dtype=np.float32)
        elements_to_matrix(
            self._descriptor_set, mat=data,
            use_multiprocessing=self.use_multiprocessing,
            report_interval=1.0,
        )
        self._uuids = np.array(list(self._descriptor_set.iterkeys()))
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

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self._descriptor_set)

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
        uuids, dists = zip(*((uuids[oidx], d_dists[oidx]) for oidx in order))

        d_dists = d_dists[order]
        self._log.debug("Min and max FAISS distances: %g, %g",
                        min(dists), max(dists))
        self._log.debug("Min and max descriptor distances: %g, %g",
                        min(d_dists), max(d_dists))

        self._log.debug("Returning query result of size %g", len(uuids))

        return (descriptors, tuple(dists))

NN_INDEX_CLASS = FaissNearestNeighborsIndex
