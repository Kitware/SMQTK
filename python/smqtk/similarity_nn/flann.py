__author__ = 'purg'

import logging
import multiprocessing
import numpy
import os
import tempfile

from smqtk.similarity_nn import SimilarityNN
from smqtk.utils import safe_create_dir

try:
    import pyflann
except ImportError:
    pyflann = None


class FlannSimilarity (SimilarityNN):
    """
    Nearest-neighbor computation using the FLANN library (pyflann module).

    This implementation uses in-memory data structures, and thus has an index
    size limit based on how much memory the running machine has available.

    NOTE: Normally, FLANN indices don't play well with multiprocessing due to
        being C structures and don't transfer into new processes memory space.
        However, FLANN can serialize an index, and so this processes uses
        temporary storage space to serialize

    """

    @classmethod
    def is_usable(cls):
        # Assuming that if the pyflann module is available, then it's going to
        # work. This assumption will probably be invalidated in the future...
        return pyflann is not None

    def __init__(self, content_descriptor, temp_dir, autotune=False,
                 target_precision=0.95, sample_fraction=0.1,
                 distance_method='chi_square'):
        """
        Initialize FLANN index properties. Index is of course not build yet (no
        data).

        Optional parameters are for when building the index. Documentation on
        their meaning can be found in the FLANN documentation PDF:

            http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf

        See the MATLAB section for detailed descriptions (python section will
        just point you to the MATLAB section).

        :param content_descriptor: Content descriptor instance to provide
            descriptors
        :type content_descriptor: smqtk.content_description.ContentDescriptor
        :param temp_dir: Directory to use for working file storage, mainly for
            saving and loading indices for multiprocess transitions
        :type temp_dir: str
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
            include "hik", "chi_square" (default), and "euclidean".
        :type distance_method: str

        """
        super(FlannSimilarity, self).__init__(content_descriptor)

        self._temp_dir = os.path.abspath(os.path.expanduser(temp_dir))

        self._build_autotune = bool(autotune)
        self._build_target_precision = float(target_precision)
        self._build_sample_frac = float(sample_fraction)

        self._distance_method = str(distance_method)

        # The flann instance with a built index. None before index construction
        #: :type: pyflann.index.FLANN or None
        self._flann = None
        # Flann index parameters determined during building. This is used when
        # re-building index when adding to it.
        #: :type: dict
        self._flann_build_params = None
        # Path to the file that is the serialization of our index. This is None
        # before index construction
        #: :type: None or str
        self._flann_index_cache = None
        # The process ID that the currently set FLANN instance was build/loaded
        # on. If this differs from the current process ID, the index should be
        # reloaded from cache.
        self._pid = None

        # Cache of descriptors we're indexing over, mapped from data elem UUID.
        # This should be preserved when forking processes.
        #: :type: list[numpy.core.multiarray.ndarray]
        self._descr_cache = None
        #: :type: list[smqtk.data_rep.DataElement]
        self._elem_cache = None
        # UUID set cache for quick existence check
        #: :type: set[collections.Hashable]
        self._uuid_cache = set()

    def _restore_index(self):
        """
        If we think we're suppose to have an index, check the recorded PID with
        the current PID, reloading the index from cache if they differ.

        If there is a loaded index and we're on the same process that created it
        this does nothing.
        """
        if self._flann_index_cache and os.path.isfile(self._flann_index_cache) \
                and self._pid != multiprocessing.current_process().pid:
            pyflann.set_distance_type(self._distance_method)
            self._flann = pyflann.FLANN()
            pts_array = numpy.array(self._descr_cache,
                                    dtype=self._descr_cache[0].dtype)
            self._flann.load_index(self._flann_index_cache, pts_array)
            self._pid = multiprocessing.current_process().pid

    def build_index(self, data):
        """
        Build the index over the given data elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        Implementation Notes:
            - We keep a cache file serialization around for our index in case
                sub-processing occurs so as to be able to recover from the
                underlying C data not being there. This could cause issues if
                a main or child process rebuild's the index, as we clear the old
                cache away.

        :param data: Iterable of data elements to build index over.
        :type data: collections.Iterable[smqtk.data_rep.DataElement]

        """
        # If there is already an index, clear the cache file if we are in the
        # same process that created our current index.
        if self._flann_index_cache and os.path.isfile(self._flann_index_cache) \
                and self._pid == multiprocessing.current_process().pid:
            self._log.debug('removing old index cache file')
            os.remove(self._flann_index_cache)

        self._log.debug("Building new index")

        # Compute descriptors for data elements
        self._log.debug("Computing descriptors for data")
        uid2vec = \
            self._content_descriptor.compute_descriptor_async(data)
        # Translate returned mapping into cache lists
        self._descr_cache = []
        self._elem_cache = []
        self._uuid_cache = set()
        for de in data:
            self._elem_cache.append(de)
            self._uuid_cache.add(de.uuid())
            self._descr_cache.append(uid2vec[de.uuid()])
        if not self._descr_cache:
            raise ValueError("No data provided in given iterable.")

        # numpy array version for FLANN
        pts_array = numpy.array(self._descr_cache,
                                dtype=self._descr_cache[0].dtype)

        # Reset PID/FLANN/saved cache
        self._pid = multiprocessing.current_process().pid
        safe_create_dir(self._temp_dir)
        fd, self._flann_index_cache = tempfile.mkstemp(".flann",
                                                       dir=self._temp_dir)
        os.close(fd)
        self._log.debug("Building FLANN index")
        pyflann.set_distance_type(self._distance_method)
        self._flann = pyflann.FLANN()
        self._flann_build_params = \
            self._flann.build_index(pts_array, **{
                "algorithm": self._build_autotune,
                "target_precision": self._build_target_precision,
                "sample_fraction": self._build_sample_frac,
                "log_level": ("info"
                              if self._log.getEffectiveLevel() <= logging.DEBUG
                              else "warn")
            })

        # Saving out index cache
        self._log.debug("Saving index to cache file: %s",
                        self._flann_index_cache)
        self._flann.save_index(self._flann_index_cache)

    def add_to_index(self, data):
        """
        Add the given data element to the index.

        :param data: New data element to add to our index.
        :type data: smqtk.data_rep.DataElement

        """
        self._restore_index()

        # If the uuid of the given data is already in the index, do nothing
        if data.uuid() in self._uuid_cache:
            self._log.debug("Index already contains data element UUID[%s]",
                            str(data.uuid()))
            return

        # add new content's descriptor to cache, rebuild FLANN index from
        # existing params, but new descriptor point set.
        vec = self._content_descriptor.compute_descriptor(data)
        self._descr_cache.append(vec)
        self._elem_cache.append(data)
        self._uuid_cache.add(data.uuid())

        pyflann.set_distance_type(self._distance_method)
        pts_array = numpy.array(self._descr_cache,
                                dtype=self._descr_cache[0].dtype)
        self._flann.build_index(pts_array, **self._flann_build_params)

    def nn(self, d, N=1):
        self._restore_index()

        if d.uuid() in self._uuid_cache:
            i = self._elem_cache.index(d)
            vec = self._descr_cache[i]
        else:
            vec = self._content_descriptor.compute_descriptor(d)

        #: :type: numpy.core.multiarray.ndarray
        idxs = self._flann.nn_index(vec, N)[0]
        # When N>1, return value is a 2D array for some reason
        if len(idxs.shape) > 1:
            idxs = idxs[0]
        return [self._elem_cache[i] for i in idxs]


SIMILARITY_NN_CLASS = FlannSimilarity
