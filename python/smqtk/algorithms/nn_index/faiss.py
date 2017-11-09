from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import json
import multiprocessing
import numpy as np
import os
import six
from six.moves import zip
import tempfile

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation import (
    get_data_element_impls,
    get_descriptor_index_impls,
)
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils import plugin, merge_dict, metrics

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

    def __init__(self, descriptor_set, index_element=None,
                 index_param_element=None,
                 read_only=False, factory_string='Flat',
                 use_multiprocessing=True,
                 random_seed=None):
        """
        Initialize FAISS index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param descriptor_set: Index in which DescriptorElements will be
            stored.
        :type descriptor_set: smqtk.representation.DescriptorIndex

        :param index_element: Optional DataElement used to load/store the
            index. When None, the index will only be stored in memory.
        :type index_element: None | smqtk.representation.DataElement

        :param index_param_element: Optional DataElement used to load/store
            the index parameters. When None, the index will only be stored in
            memory.
        :type index_param_element: None | smqtk.representation.DataElement

        :param read_only: If True, `build_index` will error if there is an
            existing index. False by default.
        :type read_only: bool

        :param factory_string: String to pass to FAISS' `index_factory`;
            see the [documentation][1] on this feature for more details.
        :type factory_string: str

        :param use_multiprocessing: Whether or not to use discrete processes
            as the parallelization agent vs python threads.
        :type use_multiprocessing: bool

        :param random_seed: Integer to use as the random number generator
            seed.
        :type random_seed: int | None

        [1]: https://github.com/facebookresearch/faiss/wiki/High-level-interface-and-auto-tuning#index-factory

        """
        super(FaissNearestNeighborsIndex, self).__init__()

        if isinstance(factory_string, six.text_type):
            self.factory_string = factory_string.encode()
        elif isinstance(factory_string, six.binary_type):
            self.factory_string = factory_string
        else:
            raise ValueError('The factory_string parameter must be a'
                             ' recognized string type.')

        self._index_element = index_element
        self._index_param_element = index_param_element
        self.read_only = read_only
        self.use_multiprocessing = use_multiprocessing
        self._descriptor_set = descriptor_set
        self._model_lock = multiprocessing.RLock()

        self.random_seed = None
        if random_seed is not None:
            self.random_seed = int(random_seed)

        # Load the index/parameters if one exists
        if self._has_model_data():
            self._log.info("Found existing model data. Loading.")
            self._load_faiss_model()
            self._uuids = list(self._descriptor_set.iterkeys())
            assert len(self._uuids) == self._faiss_index.ntotal, \
                "The number of descriptors doesn't match the number of " \
                "items in the index."

    def get_config(self):
        config = {
            "descriptor_set": plugin.to_plugin_config(self._descriptor_set),
            "factory_string": self.factory_string,
            "read_only": self.read_only,
            "random_seed": self.random_seed,
            "use_multiprocessing": self.use_multiprocessing,
        }
        if self._index_element:
            config['index_element'] = plugin.to_plugin_config(
                self._index_element)
        if self._index_param_element:
            config['index_param_element'] = plugin.to_plugin_config(
                self._index_param_element)

        return config

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

        data_element_default_config = plugin.make_config(
            get_data_element_impls())
        default['index_element'] = data_element_default_config

        data_element_default_config = plugin.make_config(
            get_data_element_impls())
        default['index_param_element'] = data_element_default_config

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

        cfg['descriptor_set'] = plugin.from_plugin_config(
            cfg['descriptor_set'], get_descriptor_index_impls())

        if (cfg['index_element'] and
                cfg['index_element']['type']):
            index_element = plugin.from_plugin_config(
                cfg['index_element'], get_data_element_impls())
            cfg['index_element'] = index_element

        if (cfg['index_param_element'] and
                cfg['index_param_element']['type']):
            index_param_element = plugin.from_plugin_config(
                cfg['index_param_element'], get_data_element_impls())
            cfg['index_param_element'] = index_param_element

        return super(FaissNearestNeighborsIndex, cls).from_config(cfg, False)

    def _has_model_data(self):
        """
        check if configured model files are configured and not empty
        """
        return (self._index_element and not self._index_element.is_empty() and
                self._index_param_element and
                not self._index_param_element.is_empty())

    def _load_faiss_model(self):
        """
        Load the FAISS model from the configured DataElement
        """
        # Load the binary index
        if self._index_element and not self._index_element.is_empty():
            tmp_fp = self._index_element.write_temp()
            self._faiss_index = faiss.read_index(tmp_fp)
            self._index_element.clean_temp()
        # Params pickle include the build params + our local state params
        if (self._index_param_element and
                not self._index_param_element.is_empty()):
            state = json.loads(self._index_param_element.get_bytes())
            self.factory_string = state["factory_string"]
            self.read_only = state["read_only"]
            self.random_seed = state["random_seed"]
            self.use_multiprocessing = state["use_multiprocessing"]

    def _save_faiss_model(self):
        """
        Save the index to the configured DataElement.
        """
        if self._index_element and self._index_element.writable():
            self._log.debug("Storing index: %s", self._index_element)
            
            # FAISS wants to write to a file, so make a temp file, then read
            # it in, putting bytes into element.
            fd, fp = tempfile.mkstemp()
            try:
                faiss.write_index(self._faiss_index, fp)
                self._index_element.set_bytes(
                    os.read(fd, os.path.getsize(fp)))
            finally:
                os.close(fd)
                os.remove(fp)
        if self._index_param_element and self._index_param_element.writable():
            params = {
                "factory_string": self.factory_string,
                "read_only": self.read_only,
                "random_seed": self.random_seed,
                "use_multiprocessing": self.use_multiprocessing,
            }
            self._index_param_element.set_bytes(json.dumps(params))

    def _build_index(self, descriptors):
        """
        Internal method to be implemented by sub-classes to build the index
        with the given descriptor data elements.

        Subsequent calls to this method should rebuild the current index.
        This method shall not add to the existing index nor raise an exception
        to as to protect the current index.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        if self.read_only:
            raise ReadOnlyError("Cannot modify read-only index.")

        self._log.info("Building new FAISS index")

        # We need to fork the iterator, so stick the elements in a list
        desc_list = list(descriptors)
        data, new_uuids = self._descriptors_to_matrix(desc_list)
        n, d = data.shape

        # Build a faiss index but don't add it until we have a lock
        faiss_index = faiss.index_factory(d, self.factory_string)
        faiss_index.train(data)
        assert faiss_index.d == d, \
            "FAISS index dimension doesn't match data dimension"

        faiss_index.add(data)
        assert faiss_index.ntotal == n, \
            "FAISS index size doesn't match data size"

        with self._model_lock:
            self._faiss_index = faiss_index
            self._log.info("FAISS index has been constructed with %d"
                           " vectors", n)

            self._log.debug("Clearing and adding new descriptor elements")
            self._descriptor_set.clear()
            self._descriptor_set.add_many_descriptors(desc_list)
            assert len(self._descriptor_set) == n, \
                "New descriptor set size doesn't match data size"

            self._uuids = new_uuids
            assert len(self._uuids) == n, \
                "New uuid list size doesn't match data size"

            self._save_faiss_model()

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
        if self.read_only:
            raise ReadOnlyError("Cannot modify read-only index.")

        if not hasattr(self, "_faiss_index"):
            self._build_index(descriptors)
            return

        self._log.debug('Updating FAISS index')

        # We need to fork the iterator, so stick the elements in a list
        desc_list = list(descriptors)
        data, new_uuids = self._descriptors_to_matrix(desc_list)
        n, d = data.shape

        with self._model_lock:
            old_ntotal = self.count()

            assert self._faiss_index.d == d, \
                "FAISS index dimension doesn't match data dimension"
            self._faiss_index.add(data)
            assert self._faiss_index.ntotal == old_ntotal + n, \
                "New FAISS index size doesn't match old + data size"
            self._log.info("FAISS index has been updated with %d"
                           " new vectors", n)

            self._log.debug("Adding new descriptor elements")
            self._descriptor_set.add_many_descriptors(desc_list)
            assert len(self._descriptor_set) == old_ntotal + n, \
                "New descriptor set size doesn't match old + data size"

            self._uuids.extend(new_uuids)
            assert len(self._uuids) == old_ntotal + n, \
                "New uuid list size doesn't match old + data size"
            
            self._save_faiss_model()

    def _descriptors_to_matrix(self, descriptors):
        """
        Extract an (n,d) array with the descriptor vectors in each row,
        and a corresponding list of uuids from the list of descriptors.

        :param descriptors: List descriptor elements to add to this
            index.
        :type descriptors: list[smqtk.representation.DescriptorElement]

        :return: An (n,d) array of descriptors (d-dim descriptors in n
            rows), and the corresponding list of descriptor uuids.
        :rtype: (np.ndarray, list[collections.Hashable])
        """
        new_uuids = [desc.uuid() for desc in descriptors]
        sample_v = descriptors[0].vector()
        n, d = len(new_uuids), sample_v.size
        data = np.empty((n, d), dtype=np.float32)
        elements_to_matrix(
            descriptors, mat=data,
            use_multiprocessing=self.use_multiprocessing,
            report_interval=1.0,
        )
        self._log.info("data shape, type: %s, %s",
                       data.shape, data.dtype)
        self._log.info("# uuids: %d", n)
        return data, new_uuids

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        with self._model_lock:
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
        q = d.vector()[np.newaxis, :].astype(np.float32)

        self._log.debug("Received query for %d nearest neighbors", n)

        with self._model_lock:
            s_dists, s_ids = self._faiss_index.search(q, n)
            s_dists, s_ids = np.sqrt(s_dists[0, :]), s_ids[0, :]
            uuids = [self._uuids[s_id] for s_id in s_ids]

            descriptors = self._descriptor_set.get_many_descriptors(uuids)

        self._log.debug("Min and max FAISS distances: %g, %g",
                        min(s_dists), max(s_dists))

        descriptors = tuple(descriptors)
        d_vectors = elements_to_matrix(descriptors)
        d_dists = metrics.euclidean_distance(d_vectors, q)

        self._log.debug("Min and max descriptor distances: %g, %g",
                        min(d_dists), max(d_dists))

        order = d_dists.argsort()
        uuids, d_dists = zip(*((uuids[oidx], d_dists[oidx]) for oidx in order))

        self._log.debug("Returning query result of size %g", len(uuids))

        return descriptors, tuple(d_dists)


NN_INDEX_CLASS = FaissNearestNeighborsIndex
