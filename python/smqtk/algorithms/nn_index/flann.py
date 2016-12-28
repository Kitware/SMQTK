import cPickle
import logging
import multiprocessing
import os
import tempfile

import numpy

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.representation.data_element import from_uri
from smqtk.representation.descriptor_element import elements_to_matrix

# Requires FLANN bindings
try:
    import pyflann
except ImportError:
    pyflann = None


class FlannNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest-neighbor computation using the FLANN library (pyflann module).

    This implementation uses in-memory data structures, and thus has an index
    size limit based on how much memory the running machine has available.

    NOTE ON MULTIPROCESSING
        Normally, FLANN indices don't play well when multiprocessing due to
        the underlying index being a C structure, which doesn't auto-magically
        transfer to forked processes like python structure data does. The
        serialized FLANN index file is used to restore a built index in separate
        processes, assuming one has been built.

    """

    @classmethod
    def is_usable(cls):
        # if underlying library is not found, the import above will error
        return pyflann is not None

    def __init__(self, index_uri=None, parameters_uri=None,
                 descriptor_cache_uri=None,
                 # Parameters for building an index
                 autotune=False, target_precision=0.95, sample_fraction=0.1,
                 distance_method='hik', random_seed=None):
        """
        Initialize FLANN index properties. Does not contain a query-able index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        When using this algorithm in a multiprocessing environment, the model
        file path parameters must be specified due to needing to reload the
        FLANN index on separate processes. This is because FLANN is in C and its
        instances are not copied into processes.

        Documentation on index building parameters and their meaning can be
        found in the FLANN documentation PDF:

            http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf

        See the MATLAB section for detailed descriptions (python section will
        just point you to the MATLAB section).

        :param index_uri: Optional URI to where to load/store FLANN index
            when initialized and/or built.

            If not configured, no model files are written to or loaded from
            disk.
        :type index_uri: None | str

        :param parameters_uri: Optional file location to load/save FLANN
            index parameters determined at build time.

            If not configured, no model files are written to or loaded from
            disk.
        :type parameters_uri: None | str

        :param descriptor_cache_uri: Optional file location to load/store
            DescriptorElements in this index.

            If not configured, no model files are written to or loaded from
            disk.
        :type descriptor_cache_uri: None | str

        :param autotune: Whether or not to perform parameter auto-tuning when
            building the index. If this is False, then the `target_precision`
            and `sample_fraction` parameters are not used.
        :type autotune: bool

        :param target_precision: Target estimation accuracy when determining
            nearest neighbor when tuning parameters. This should be between
            [0,1] and represents percentage accuracy.
        :type target_precision: float

        :param sample_fraction: Sub-sample percentage of the total index to use
            when performing auto-tuning. Value should be in the range of [0,1]
            and represents percentage.
        :type sample_fraction: float

        :param distance_method: Method label of the distance function to use.
            See FLANN documentation manual for available methods. Common methods
            include "hik", "chi_square" (default), and "euclidean". When loading
            and existing index, this value is ignored in preference for the
            distance method used to build the loaded index.
        :type distance_method: str

        :param random_seed: Integer to use as the random number generator seed.
        :type random_seed: int

        """
        super(FlannNearestNeighborsIndex, self).__init__()

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

        # parameters for building an index
        self._build_autotune = autotune
        self._build_target_precision = float(target_precision)
        self._build_sample_frac = float(sample_fraction)

        self._distance_method = str(distance_method)

        # In-order cache of descriptors we're indexing over.
        # - flann.nn_index will spit out indices to list
        #: :type: list[smqtk.representation.DescriptorElement] | None
        self._descr_cache = None

        # The flann instance with a built index. None before index load/build.
        #: :type: pyflann.index.FLANN or None
        self._flann = None
        # Flann index parameters determined during building. None before index
        # load/build.
        #: :type: dict
        self._flann_build_params = None

        #: :type: None | int
        self._rand_seed = None
        if random_seed:
            self._rand_seed = int(random_seed)

        # The process ID that the currently set FLANN instance was built/loaded
        # on. If this differs from the current process ID, the index should be
        # reloaded from cache.
        self._pid = None

        # Load the index/parameters if one exists
        if self._has_model_data():
            self._log.info("Found existing model data. Loading.")
            self._load_flann_model()

    def get_config(self):
        return {
            "index_uri": self._index_uri,
            "parameters_uri": self._index_param_uri,
            "descriptor_cache_uri": self._descr_cache_uri,
            "autotune": self._build_autotune,
            "target_precision": self._build_target_precision,
            "sample_fraction": self._build_sample_frac,
            "distance_method": self._distance_method,
            "random_seed": self._rand_seed,
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

        # Params pickle include the build params + our local state params
        if self._index_param_elem and not self._index_param_elem.is_empty():
            state = cPickle.loads(self._index_param_elem.get_bytes())
            self._build_autotune = state['b_autotune']
            self._build_target_precision = state['b_target_precision']
            self._build_sample_frac = state['b_sample_frac']
            self._distance_method = state['distance_method']
            self._flann_build_params = state['flann_build_params']

        # Load the binary index
        if self._index_elem and not self._index_elem.is_empty():
            # make numpy matrix of descriptor vectors for FLANN
            pts_array = [d.vector() for d in self._descr_cache]
            pts_array = numpy.array(pts_array, dtype=pts_array[0].dtype)
            pyflann.set_distance_type(self._distance_method)
            self._flann = pyflann.FLANN()
            tmp_fp = self._index_elem.write_temp()
            self._flann.load_index(tmp_fp, pts_array)
            self._index_elem.clean_temp()
            del pts_array, tmp_fp

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

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self._descr_cache) if self._descr_cache else 0

    def build_index(self, descriptors):
        """
        Build the index over the descriptors data elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        Implementation Notes:
            - We keep a cache file serialization around for our index in case
                sub-processing occurs so as to be able to recover from the
                underlying C data not being there. This could cause issues if
                a main or child process rebuild's the index, as we clear the old
                cache away.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptors elements to build index
            over.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        # Not caring about restoring the index because we're just making a new
        # one
        self._log.info("Building new FLANN index")

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

        params = {
            "target_precision": self._build_target_precision,
            "sample_fraction": self._build_sample_frac,
            "log_level": ("info"
                          if self._log.getEffectiveLevel() <= logging.DEBUG
                          else "warning")
        }
        if self._build_autotune:
            params['algorithm'] = "autotuned"
        if self._rand_seed is not None:
            params['random_seed'] = self._rand_seed
        pyflann.set_distance_type(self._distance_method)

        self._log.debug("Accumulating descriptor vectors into matrix for FLANN")
        pts_array = elements_to_matrix(self._descr_cache, report_interval=1.0)

        self._log.debug('Building FLANN index')
        self._flann = pyflann.FLANN()
        self._flann_build_params = self._flann.build_index(pts_array, **params)
        del pts_array

        if self._index_elem and self._index_elem.writable():
            self._log.debug("Caching index: %s", self._index_elem)
            # FLANN wants to write to a file, so make a temp file, then read it
            # in, putting bytes into element.
            fd, fp = tempfile.mkstemp()
            try:
                self._flann.save_index(fp)
                self._index_elem.set_bytes(os.read(fd, os.path.getsize(fp)))
            finally:
                os.close(fd)
                os.remove(fp)
        if self._index_param_elem and self._index_param_elem.writable():
            self._log.debug("Caching index params: %s", self._index_param_elem)
            state = {
                'b_autotune': self._build_autotune,
                'b_target_precision': self._build_target_precision,
                'b_sample_frac': self._build_sample_frac,
                'distance_method': self._distance_method,
                'flann_build_params': self._flann_build_params,
            }
            self._index_param_elem.set_bytes(cPickle.dumps(state, -1))

        self._pid = multiprocessing.current_process().pid

    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """
        self._restore_index()
        super(FlannNearestNeighborsIndex, self).nn(d, n)
        vec = d.vector()

        # If the distance method is HIK, we need to treat it special since that
        # method produces a similarity score, not a distance score.
        #
        # FLANN asserts that we query for <= index size, thus the use of min()
        if self._distance_method == 'hik':
            # This call is different than the else version in that k is the size
            # of the full data set, so that we can reverse the distances
            #: :type: numpy.ndarray, numpy.ndarray
            idxs, dists = self._flann.nn_index(vec, len(self._descr_cache),
                                               **self._flann_build_params)
        else:
            #: :type: numpy.ndarray, numpy.ndarray
            idxs, dists = self._flann.nn_index(vec,
                                               min(n, len(self._descr_cache)),
                                               **self._flann_build_params)

        # When N>1, return value is a 2D array. Since this method limits query
        #   to a single descriptor, we reduce to 1D arrays.
        if len(idxs.shape) > 1:
            idxs = idxs[0]
            dists = dists[0]

        if self._distance_method == 'hik':
            # Invert values to stay consistent with other distance value norms
            dists = [1.0 - d for d in dists]
            idxs = tuple(reversed(idxs))[:n]
            dists = tuple(reversed(dists))[:n]
        else:
            dists = tuple(dists)

        return [self._descr_cache[i] for i in idxs], dists


NN_INDEX_CLASS = FlannNearestNeighborsIndex
