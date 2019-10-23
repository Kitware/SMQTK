from __future__ import absolute_import, division, print_function

import collections
from copy import deepcopy
import json
import multiprocessing
import numpy as np
import os
import six
import tempfile
import warnings

from six.moves import zip, filter

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation import (
    DataElement,
    DescriptorSet,
    KeyValueStore,
)
from smqtk.representation.descriptor_element import DescriptorElement
from smqtk.utils import metrics
from smqtk.utils.configuration import \
    make_default_config, from_config_dict, to_config_dict
from smqtk.utils.dict import merge_dict

# Requires FAISS bindings
try:
    import faiss
except ImportError as ex:
    warnings.warn("FaissNearestNeighborsIndex is not usable due to the faiss "
                  "module not being importable: {}".format(str(ex)))
    faiss = None


# TODO: Add metric constructor option, append to ``faiss.METRIC_{}`` for
#       library constant.
# TODO: Add flag for optional memory mapping of index file.
# TODO: Options to "train" with up to configured N descriptors instead of
#       everything (maybe too many input).
# TODO: Add parameter for updating index in batches for when loading all
#       descriptors as a matrix into memory is too much.


class FaissNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest-neighbor computation using the FAISS library.
    """

    @staticmethod
    def gpu_supported():
        """
        :return: If FAISS seems to have GPU support or not.
        :rtype: bool
        """
        # Test if the GPU version is available
        if hasattr(faiss, "StandardGpuResources"):
            return True
        else:
            return False

    @classmethod
    def is_usable(cls):
        # if underlying library is not found, the import above will error
        return faiss is not None

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

        data_element_default_config = \
            make_default_config(DataElement.get_impls())
        default['index_element'] = data_element_default_config
        default['index_param_element'] = deepcopy(data_element_default_config)

        di_default = make_default_config(DescriptorSet.get_impls())
        default['descriptor_set'] = di_default

        kvs_default = make_default_config(KeyValueStore.get_impls())
        default['idx2uid_kvs'] = kvs_default
        default['uid2idx_kvs'] = deepcopy(kvs_default)

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

        cfg['descriptor_set'] = from_config_dict(
            cfg['descriptor_set'], DescriptorSet.get_impls()
        )
        cfg['uid2idx_kvs'] = from_config_dict(
            cfg['uid2idx_kvs'], KeyValueStore.get_impls()
        )
        cfg['idx2uid_kvs'] = from_config_dict(
            cfg['idx2uid_kvs'], KeyValueStore.get_impls()
        )

        if (cfg['index_element'] and
                cfg['index_element']['type']):
            index_element = from_config_dict(
                cfg['index_element'], DataElement.get_impls())
            cfg['index_element'] = index_element
        else:
            cfg['index_element'] = None

        if (cfg['index_param_element'] and
                cfg['index_param_element']['type']):
            index_param_element = from_config_dict(
                cfg['index_param_element'], DataElement.get_impls())
            cfg['index_param_element'] = index_param_element
        else:
            cfg['index_param_element'] = None

        return super(FaissNearestNeighborsIndex, cls).from_config(cfg, False)

    def __init__(self, descriptor_set, idx2uid_kvs, uid2idx_kvs,
                 index_element=None, index_param_element=None,
                 read_only=False, factory_string='IDMap,Flat',
                 ivf_nprobe=1, use_gpu=False, gpu_id=0, random_seed=None):
        """
        Initialize FAISS index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param descriptor_set: Set in which indexed DescriptorElements will be
            stored.
        :type descriptor_set: smqtk.representation.DescriptorSet

        :param idx2uid_kvs: Key-value storage mapping FAISS indexed vector
            index to descriptor UID.  This should be the inverse of
            `uid2idx_kvs`.
        :type idx2uid_kvs: smqtk.representation.KeyValueStore

        :param uid2idx_kvs: Key-value storage mapping descriptor UIDs to FAISS
            indexed vector index.  This should be the inverse of `idx2uid_kvs`.
        :type uid2idx_kvs: smqtk.representation.KeyValueStore

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
            see the documentation [1] on this feature for more details.
        :type factory_string: str | unicode

        :param int ivf_nprobe:
            If an IVF-type index is loaded, optionally use this as the
            ``nprobe`` value at query time.
            This should be an integer greater than 1 to be effective.
            This parameter is ignore if the loaded index is not IVF-based.
            When this is None the FAISS IVF default value for ``nprobe`` is
            used (1).

        :param use_gpu: Use a GPU index if GPU support is available.  A
            RuntimeError is thrown during instance construction if GPU support
            is not available and this flag is true.  See the following for
            FAISS GPU documentation and limitations:

                https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
        :type use_gpu: bool

        :param gpu_id: If the GPU implementation is available for FAISS
            (automatically determined) use the GPU with this device number /
            ID.
        :type gpu_id: int

        :param random_seed: Integer to use as the random number generator
            seed.
        :type random_seed: int | None

        [1]: https://github.com/facebookresearch/faiss/wiki/High-level-interface-and-auto-tuning#index-factory

        """
        super(FaissNearestNeighborsIndex, self).__init__()

        if not isinstance(factory_string, six.string_types):
            raise ValueError('The factory_string parameter must be a '
                             'recognized string type.')

        self._descriptor_set = descriptor_set
        self._idx2uid_kvs = idx2uid_kvs
        self._uid2idx_kvs = uid2idx_kvs
        self._index_element = index_element
        self._index_param_element = index_param_element
        self.read_only = read_only
        self.factory_string = str(factory_string)
        self._ivf_nprobe = int(ivf_nprobe)
        if self._ivf_nprobe < 1:
            raise ValueError("ivf_nprobe must be >= 1.")
        self._use_gpu = use_gpu
        self._gpu_id = gpu_id
        self.random_seed = None
        if random_seed is not None:
            self.random_seed = int(random_seed)
        # Index value for the next added element.  Reset to 0 on a build.
        self._next_index = 0

        # Place-holder for option GPU resource reference. Just exist for the
        # duration of the index converted with it.
        self._gpu_resources = None
        if self._use_gpu and not self.gpu_supported():
            raise RuntimeError("Requested GPU use but FAISS does not seem to "
                               "support GPU functionality.")

        # Lock for accessing FAISS model components.
        # - GPU index access is NOT thread-safe
        #   https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls#thread-safety
        self._model_lock = multiprocessing.RLock()
        # Placeholder for FAISS model instance.
        #: :type: None | faiss.Index
        self._faiss_index = None

        # Load the index/parameters if one exists
        self._load_faiss_model()

    def get_config(self):
        config = {
            "descriptor_set": to_config_dict(self._descriptor_set),
            "uid2idx_kvs": to_config_dict(self._uid2idx_kvs),
            "idx2uid_kvs": to_config_dict(self._idx2uid_kvs),
            "factory_string": self.factory_string,
            "ivf_nprobe": self._ivf_nprobe,
            "read_only": self.read_only,
            "random_seed": self.random_seed,
            "use_gpu": self._use_gpu,
            "gpu_id": self._gpu_id,
        }
        if self._index_element:
            config['index_element'] = to_config_dict(
                self._index_element)
        if self._index_param_element:
            config['index_param_element'] = to_config_dict(
                self._index_param_element)

        return config

    def _convert_index(self, faiss_index):
        """
        Convert the given index to a GpuIndex if `use_gpu` is True, otherwise
        return the index given (no-op).

        :param faiss_index: Index to convert.
        :type faiss_index: faiss.Index

        :return: Optionally converted index.
        :rtype: faiss.Index | faiss.GpuIndex

        """
        # If we're to use a GPU index and what we're given isn't already a GPU
        # index.
        if self._use_gpu and not isinstance(faiss_index, faiss.GpuIndex):
            self._log.debug("-> GPU-enabling index")
            # New resources
            self._gpu_resources = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(),
                                                 self._gpu_id, faiss_index)
        return faiss_index

    def _index_factory_wrapper(self, d, factory_string):
        """
        Create a FAISS index for the given descriptor dimensionality and
        factory string.

        This *always* produces an index using the L2 metric.

        :param d: Integer indexed vector dimensionality.
        :type d: int

        :param factory_string: Factory string to drive index generation.
        :type factory_string: str

        :return: Constructed index.
        :rtype: faiss.Index | faiss.GpuIndex
        """
        self._log.debug("Creating index by factory: '%s'", factory_string)
        index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)
        return self._convert_index(index)

    def _has_model_data(self):
        """
        Check if configured model files are configured and not empty.
        """
        with self._model_lock:
            return (self._index_element and
                    self._index_param_element and
                    not self._index_element.is_empty() and
                    not self._index_param_element.is_empty())

    def _load_faiss_model(self):
        """
        Load the FAISS model from the configured DataElement
        """
        with self._model_lock:
            if self._has_model_data():
                # Load the binary index
                tmp_fp = self._index_element.write_temp()
                self._faiss_index = self._convert_index(
                    # As of Faiss 1.3.0, only str (not unicode) is
                    # accepted in Python 2.7
                    faiss.read_index(str(tmp_fp))
                )
                self._index_element.clean_temp()

                # Params pickle include the build params + our local state
                # params.
                state = json.loads(self._index_param_element.get_bytes())
                self.factory_string = state["factory_string"]
                self.read_only = state["read_only"]
                self.random_seed = state["random_seed"]
                self._next_index = state["next_index"]

                # Check that descriptor-set and kvstore instances match up in
                # size.
                if not (
                        len(self._descriptor_set) == len(self._uid2idx_kvs) ==
                        len(self._idx2uid_kvs) == self._faiss_index.ntotal):
                    self._log.warning(
                        "Not all of our storage elements agree on size: "
                        "len(dset, uid2idx, idx2uid, faiss_idx) = "
                        "(%d, %d, %d, %d)"
                        % (len(self._descriptor_set), len(self._uid2idx_kvs),
                           len(self._idx2uid_kvs), self._faiss_index.ntotal)
                    )

    def _save_faiss_model(self):
        """
        Save the index and parameters to the configured DataElements.
        """
        with self._model_lock:
            # Only write to cache elements if they are both writable.
            writable = (self._index_element and
                        self._index_element.writable() and
                        self._index_param_element and
                        self._index_param_element.writable())
            if writable:
                self._log.debug("Storing index: %s", self._index_element)
                # FAISS wants to write to a file, so make a temp file, then
                # read it in, putting bytes into element.
                fd, fp = tempfile.mkstemp()
                try:
                    # Write function needs a CPU index instance, so bring it
                    # down from the GPU if necessary.
                    if self._use_gpu and isinstance(self._faiss_index,
                                                    faiss.GpuIndex):
                        to_write = faiss.index_gpu_to_cpu(self._faiss_index)
                    else:
                        to_write = self._faiss_index
                    faiss.write_index(to_write, fp)
                    # Use the file descriptor to create the file object.
                    # This avoids reopening the file and will automatically
                    # close the file descriptor on exiting the with block.
                    # fdopen() is required because in Python 2 open() does
                    # not accept a file descriptor.
                    with os.fdopen(fd, 'rb') as f:
                        self._index_element.set_bytes(f.read())
                finally:
                    os.remove(fp)
                # Store index parameters used.
                params = {
                    "factory_string": self.factory_string,
                    "read_only": self.read_only,
                    "random_seed": self.random_seed,
                    "next_index": self._next_index,
                }
                # Using UTF-8 due to recommendation (of either 8, 16 or 32) by
                # the ``json.loads`` method documentation.
                self._index_param_element.set_bytes(
                    json.dumps(params).encode("utf-8")
                )

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
        idx_ids = np.arange(n)  # restart IDs from 0.

        # Build a faiss index but don't internalize it until we have a lock.

        faiss_index = self._index_factory_wrapper(d, self.factory_string)
        self._log.info("Training FAISS index")
        # noinspection PyArgumentList
        faiss_index.train(data)
        # TODO(john.moeller): This will raise an exception on flat indexes.
        # There's a solution which involves wrapping the index in an
        # IndexIDMap, but it doesn't work because of a bug in FAISS. So for
        # now we don't support flat indexes.
        self._log.info("Adding data to index")
        # noinspection PyArgumentList
        faiss_index.add_with_ids(data, idx_ids)

        assert faiss_index.d == d, \
            "FAISS index dimension doesn't match data dimension"
        assert faiss_index.ntotal == n, \
            "FAISS index size doesn't match data size"

        with self._model_lock:
            self._faiss_index = faiss_index
            self._log.info("FAISS index has been constructed with %d "
                           "vectors", n)

            self._log.debug("Clearing and adding new descriptor elements")
            self._descriptor_set.clear()
            self._descriptor_set.add_many_descriptors(desc_list)
            assert len(self._descriptor_set) == n, \
                "New descriptor set size doesn't match data size"
            idx_ids = idx_ids.astype(object)

            self._uid2idx_kvs.clear()
            self._uid2idx_kvs.add_many(
                dict(zip(new_uuids, idx_ids))
            )
            assert len(self._uid2idx_kvs) == n, \
                "New uid2idx map size doesn't match data size."

            self._idx2uid_kvs.clear()
            self._idx2uid_kvs.add_many(
                dict(zip(idx_ids, new_uuids))
            )
            assert len(self._idx2uid_kvs) == n, \
                "New idx2uid map size doesn't match data size."

            self._next_index = n

            self._save_faiss_model()

    def _update_index(self, descriptors):
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more descriptor elements given.

        If no index exists yet, a new one should be created using the given
        descriptors.

        If any descriptors have already been added, they will be not be
        re-inserted, but a warning will be raised.

        :param descriptors: Iterable of descriptor elements to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        if self.read_only:
            raise ReadOnlyError("Cannot modify read-only index.")

        if self._faiss_index is None:
            self._build_index(descriptors)
            return

        self._log.debug('Updating FAISS index')

        with self._model_lock:
            # Remove any uids which have already been indexed. This gracefully
            # handles the unusual case that the underlying FAISS index and the
            # SMQTK descriptor set have fallen out of sync due to an unexpected
            # external failure.
            desc_list = []
            for descriptor_ in descriptors:
                if descriptor_.uuid() in self._uid2idx_kvs:
                    warnings.warn(
                        "Descriptor with UID {} already present in this"
                        " index".format(descriptor_.uuid())
                    )
                else:
                    desc_list.append(descriptor_)
            data, new_uuids = self._descriptors_to_matrix(desc_list)

            n, d = data.shape

            old_ntotal = self.count()

            next_next_index = self._next_index + n
            new_ids = np.arange(self._next_index, next_next_index)
            self._next_index = next_next_index

            assert self._faiss_index.d == d, \
                "FAISS index dimension doesn't match data dimension"
            # noinspection PyArgumentList
            self._faiss_index.add_with_ids(data, new_ids)
            assert self._faiss_index.ntotal == old_ntotal + n, \
                "New FAISS index size doesn't match old + data size"
            self._log.info("FAISS index has been updated with %d"
                           " new vectors", n)

            self._log.debug("Adding new descriptor elements")
            self._descriptor_set.add_many_descriptors(desc_list)
            assert len(self._descriptor_set) == old_ntotal + n, \
                "New descriptor set size doesn't match old + data size"

            new_ids = new_ids.astype(object)

            self._uid2idx_kvs.add_many(
                dict(zip(new_uuids, new_ids))
            )
            assert len(self._uid2idx_kvs) == old_ntotal + n, \
                "New uid2idx kvs size doesn't match old + new data size."

            self._idx2uid_kvs.add_many(
                dict(zip(new_ids, new_uuids))
            )
            assert len(self._idx2uid_kvs) == old_ntotal + n, \
                "New idx2uid kvs size doesn't match old + new data size."

            self._save_faiss_model()

    def _remove_from_index(self, uids):
        """
        Internal method to be implemented by sub-classes to partially remove
        descriptors from this index associated with the given UIDs.

        :param uids: Iterable of UIDs of descriptors to remove from this index.
        :type uids: collections.Iterable[collections.Hashable]

        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.

        """
        if self.read_only:
            raise ReadOnlyError("Cannot modify read-only index.")

        with self._model_lock:
            # Check that provided IDs are present in uid2idx mapping.
            uids_d = collections.deque()
            for uid in uids:
                if uid not in self._uid2idx_kvs:
                    raise KeyError(uid)
                uids_d.append(uid)

            # Remove elements from structures
            # - faiss remove_ids requires a np.ndarray of int64 type.
            rm_idxs = np.asarray([self._uid2idx_kvs[uid] for uid in uids_d],
                                 dtype=np.int64)
            self._faiss_index.remove_ids(rm_idxs)
            self._descriptor_set.remove_many_descriptors(uids_d)
            self._uid2idx_kvs.remove_many(uids_d)
            self._idx2uid_kvs.remove_many(rm_idxs)
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
        data = np.vstack(
            DescriptorElement.get_many_vectors(descriptors)
        ).astype(np.float32)
        self._log.info("data shape, type: %s, %s",
                       data.shape, data.dtype)
        self._log.info("# uuids: %d", len(new_uuids))
        return data, new_uuids

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        with self._model_lock:
            # If we don't have a searchable index we don't actually have
            # anything.
            if self._faiss_index:
                return self._faiss_index.ntotal
            else:
                return 0

    def _idx_has_nprobe(self):
        """
        Determine if the current index supports setting the nprobe parameter
        :return:
        """

    def _set_index_nprobe(self):
        """
        Try to set the currently configured nprobe value to the current faiss
        index.

        :returns: True if nprobe was actually set and False if it wasn't (not
            an appropriate index type).
        """
        with self._model_lock:
            idx = self._faiss_index
            idx_name = idx.__class__.__name__
            try:
                # Attempting to use GpuParameterSpace doesn't error and seems
                # to function even when there is no GPU available, so the usual
                # pythonic EAFP doesn't cause an exception to catch when doing
                # the "improper" thing first.
                if isinstance(idx, faiss.GpuIndex):
                    ps = faiss.GpuParameterSpace()
                else:
                    ps = faiss.ParameterSpace()
                ps.set_index_parameter(
                    idx, 'nprobe', self._ivf_nprobe
                )
                self._log.debug("Set nprobe={} to index, instance of {}"
                                .format(self._ivf_nprobe, idx_name))
                return True
            except RuntimeError as sip_ex:
                s_ex = str(sip_ex)
                if "could not set parameter nprobe" in s_ex:
                    # OK, index does not support nprobe parameter
                    self._log.debug("Current index ({}) does not support the "
                                    "nprobe parameter."
                                    .format(idx_name))
                    return False
                # Otherwise re-raise
                raise

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
        log = self._log
        q = d.vector()[np.newaxis, :].astype(np.float32)
        log.debug("Received query for %d nearest neighbors", n)

        with self._model_lock:
            # Attempt to set n-probe of an IVF index
            self._set_index_nprobe()

            # noinspection PyArgumentList
            s_dists, s_ids = self._faiss_index.search(
                q, k=min(n, self._faiss_index.ntotal)
            )
            s_dists, s_ids = np.sqrt(s_dists[0, :]), s_ids[0, :]
            s_ids = s_ids.astype(object)
            # s_id (the FAISS index indices) can equal -1 if fewer than the
            # requested number of nearest neighbors is returned. In this case,
            # eliminate the -1 entries
            self._log.debug("Getting descriptor UIDs from idx2uid mapping.")
            uuids = list(self._idx2uid_kvs.get_many(
                filter(lambda s_id_: s_id_ >= 0, s_ids)
            ))
            if len(uuids) < n:
                warnings.warn("Less than n={} neighbors were retrieved from "
                              "the FAISS index instance. Maybe increase "
                              "nprobe if this is an IVF index?"
                              .format(n), RuntimeWarning)

            descriptors = tuple(
                self._descriptor_set.get_many_descriptors(uuids)
            )

        log.debug("Min and max FAISS distances: %g, %g",
                  min(s_dists), max(s_dists))

        d_vectors = np.vstack(
            DescriptorElement.get_many_vectors(descriptors)
        )
        d_dists = metrics.euclidean_distance(d_vectors, q)

        log.debug("Min and max descriptor distances: %g, %g",
                  min(d_dists), max(d_dists))

        order = d_dists.argsort()
        uuids, d_dists = zip(*((uuids[oidx], d_dists[oidx]) for oidx in order))

        log.debug("Returning query result of size %g", len(uuids))

        return descriptors, tuple(d_dists)


SMQTK_PLUGIN_CLASS = FaissNearestNeighborsIndex

