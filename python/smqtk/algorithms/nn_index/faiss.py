from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
from multiprocessing import RLock
import six
from six.moves import cPickle as pickle, zip

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation import get_descriptor_index_impls
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import plugin, merge_dict


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

    def __init__(self, descriptor_set, read_only=False,
                 factory_string=b'Flat',
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

        :param factory_string: String to pass to FAISS' `index_factory`;
            see the
            [documentation](https://github.com/facebookresearch/faiss/wiki/High-level-interface-and-auto-tuning#index-factory)
            on this feature for more details.
        :type factory_string: six.binary_type

        :param use_multiprocessing: Whether or not to use discrete processes
            as the parallelization agent vs python threads.
        :type use_multiprocessing: bool

        :param pickle_protocol: The protocol version to be used by the pickle
            module to serialize class information
        :type pickle_protocol: int

        :param random_seed: Integer to use as the random number generator
            seed.
        :type random_seed: int

        """
        super(FaissNearestNeighborsIndex, self).__init__()

        if isinstance(factory_string, six.text_type):
            self.factory_string = factory_string.encode()
        elif isinstance(factory_string, six.binary_type):
            self.factory_string = factory_string
        else:
            raise ValueError('The factory_string parameter must be a'
                             ' recognized string type.')

        self.read_only = read_only
        self.use_multiprocessing = use_multiprocessing
        self.pickle_protocol = pickle_protocol
        self._descriptor_set = descriptor_set
        self._lock = RLock()

        self.random_seed = None
        if random_seed is not None:
            self.random_seed = int(random_seed)

    def get_config(self):
        return {
            "descriptor_set": plugin.to_plugin_config(self._descriptor_set),
            "factory_string": self.factory_string,
            "read_only": self.read_only,
            "random_seed": self.random_seed,
            "pickle_protocol": self.pickle_protocol,
            "use_multiprocessing": self.use_multiprocessing,
        }

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as
        arguments, turning those argument names into configuration dictionary
        keys. If any of those arguments have defaults, we will add those
        values into the configuration dictionary appropriately. The dictionary
        returned should only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this
        class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        default = super(FaissNearestNeighborsIndex, cls).get_default_config()

        di_default = plugin.make_config(get_descriptor_index_impls())
        default['descriptor_set'] = di_default

        return default

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

        if isinstance(cfg['factory_string'], six.text_type):
            cfg['factory_string'] = cfg['factory_string'].encode()
        elif not isinstance(cfg['factory_string'], six.binary_type):
            raise ValueError('The factory_string parameter must be a'
                             ' recognized string type.')

        cfg['descriptor_set'] = plugin.from_plugin_config(
            cfg['descriptor_set'], get_descriptor_index_impls())

        return super(FaissNearestNeighborsIndex, cls).from_config(cfg, False)

    def _build_index(self, descriptors):
        """
        Internal method to be implemented by sub-classes to build the index with
        the given descriptor data elements.

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        if self.read_only:
            raise ReadOnlyError("Cannot modify container attributes due to "
                                "being in read-only mode.")

        self._log.info("Building new FAISS index")

        self._log.debug("Clearing and adding new descriptor elements")
        with self._lock:
            self._descriptor_set.clear()
            self._descriptor_set.add_many_descriptors(descriptors)
            self._build_faiss_model()

    def _update_index(self, descriptors):
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more descriptor elements given.

        If no index exists yet, a new one should be created using the given
        descriptors.

        :param descriptors: Iterable of descriptor elements to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        with self._lock:
            # This experimental wrapper does not currently plug into FAISS's
            # update capabilities.
            raise NotImplementedError()

    def _build_faiss_model(self):
        self._log.debug('Building FAISS index')
        with self._lock:
            sample = next(self._descriptor_set.iterdescriptors())
            sample_v = sample.vector()
            n, d = self.count(), sample_v.size

            data = np.empty((n, d), dtype=np.float32)
            elements_to_matrix(
                self._descriptor_set, mat=data,
                use_multiprocessing=self.use_multiprocessing,
                report_interval=1.0,
            )
            self._uuids = list(self._descriptor_set.iterkeys())

            self._faiss_index = faiss.index_factory(d, self.factory_string)

            self._log.info("data shape, type: %s, %s",
                           data.shape, data.dtype)
            self._log.info("# uuids: %d", len(self._uuids))
            self._faiss_index.train(data)
            self._faiss_index.add(data)

            self._log.info("FAISS index has been constructed with %d"
                           " vectors", self._faiss_index.ntotal)

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        with self._lock:
            return len(self._descriptor_set)

    def _nn(self, d, n=1):
        """
        Internal method to be implemented by sub-classes to return the nearest
        `N` neighbors to the given descriptor element.

        When this internal method is called, we have already checked that there
        is a vector in ``d`` and our index is not empty.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """
        q = d.vector().reshape(1, -1).astype(np.float32)

        self._log.debug("Received query for %d nearest neighbors", n)

        with self._lock:
            dists, ids = self._faiss_index.search(q, n)
            dists, ids = np.sqrt(dists[0,:]), ids[0,:]
            uuids = [self._uuids[id] for id in ids]

            descriptors = self._descriptor_set.get_many_descriptors(uuids)

        descriptors = tuple(descriptors)
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

        return descriptors, tuple(d_dists)

NN_INDEX_CLASS = FaissNearestNeighborsIndex
