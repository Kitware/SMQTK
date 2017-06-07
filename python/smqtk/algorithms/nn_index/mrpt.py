from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import six.moves.cPickle as pickle
from six.moves import range

# import logging
import multiprocessing
import os.path as osp

import numpy as np
from scipy.sparse import csr_matrix, random
from scipy import stats

from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils.file_utils import safe_create_dir


__author__ = b"john.moeller@kitware.com"


class MRPTNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest Neighbors index that uses the MRPT algorithm of [url]
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, index_filepath=None, parameters_filepath=None,
                 descriptor_cache_filepath=None,
                 # Parameters for building an index
                 num_trees=1, depth=1, required_votes=1, random_seed=None):
        """
        Initialize MRPT index properties. Does not contain a query-able index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        When using this algorithm in a multiprocessing environment, the model
        file path parameters must be specified due to needing to reload the
        index on separate processes.

        :param index_filepath: Optional file location to load/store MRPT index
            when initialized and/or built.

            If not configured, no model files are written to or loaded from
            disk.
        :type index_filepath: None | str

        :param parameters_filepath: Optional file location to load/save index
            parameters determined at build time.

            If not configured, no model files are written to or loaded from
            disk.
        :type parameters_filepath: None | str

        :param descriptor_cache_filepath: Optional file location to load/store
            DescriptorElements in this index.

            If not configured, no model files are written to or loaded from
            disk.
        :type descriptor_cache_filepath: None | str

        :param num_trees: The number of trees that will be generated for the
            data structure
        :type num_trees: None | int

        :param depth: The depth of the trees
        :type depth: None | int

        :param required_votes: The number of votes required for a nearest
            neighbor
        :type required_votes: None | int

        :param random_seed: Integer to use as the random number generator seed.
        :type random_seed: int

        """
        super(MRPTNearestNeighborsIndex, self).__init__()

        def normpath(p):
            return (p and osp.abspath(osp.expanduser(p))) or p

        self._index_filepath = normpath(index_filepath)
        self._index_param_filepath = normpath(parameters_filepath)
        self._descr_cache_filepath = normpath(descriptor_cache_filepath)
        # Now they're either None or an absolute path

        # parameters for building an index
        self._depth = depth
        self._num_trees = num_trees
        self._required_votes = required_votes

        self._trees = []

        # In-order cache of descriptors we're indexing over.
        # - index will spit out indices to list
        #: :type: list[smqtk.representation.DescriptorElement] | None
        self._descr_cache = None

        #: :type: None | int
        self._rand_seed = None
        if random_seed:
            self._rand_seed = int(random_seed)

        # The process ID that the currently set MRPT instance was built/loaded
        # on. If this differs from the current process ID, the index should be
        # reloaded from cache.
        self._pid = None

        # Load the index/parameters if one exists
        if self._has_model_files():
            self._log.info("Found existing model files. Loading.")
            self._load_mrpt_model()

    def get_config(self):
        return {
            "index_filepath": self._index_filepath,
            "parameters_filepath": self._index_param_filepath,
            "descriptor_cache_filepath": self._descr_cache_filepath,
            "random_seed": self._rand_seed,
            "depth": self._depth,
            "num_trees": self._num_trees,
            "required_votes": self._required_votes,
        }

    def _has_model_files(self):
        """
        check if configured model files are configured and exist
        """
        return (self._index_filepath and osp.isfile(self._index_filepath) and
                self._index_param_filepath and osp.isfile(self._index_param_filepath) and
                self._descr_cache_filepath and osp.isfile(self._descr_cache_filepath))

    def _build_multiple_trees(self, pts):
        """
        Build an MRPT structure for data pts
        :param pts: The data. Each row is a datum.
        :type pts: np.ndarray
        """
        n, d = pts.shape

        # Get the Normal distribution RNG
        rvs = stats.norm().rvs
        # Start with no trees
        self._trees = []
        # 1/sqrt(depth) considered optimal for random projections
        density = 1 / np.sqrt(self._depth)
        for _ in range(self._num_trees):
            # Each tree has a basis of sparse random projections
            random_basis = random(d, self._depth, density=density, format=b"csc",
                                  dtype=np.float64, random_state=self._rand_seed,
                                  data_rvs=rvs)
            # Array of splits is a packed tree
            splits = np.empty(((1 << self._depth) - 1,), np.float64)

            # Build the tree & store it
            leaves = self._build_single_tree(pts * random_basis, np.arange(n), splits)
            self._trees.append({
                'random_basis': random_basis,
                'splits': splits,
                'leaves': leaves
            })

    def _build_single_tree(self, proj, indices, splits, split_index=0, level=0):
        """
        Build a single RP tree for fast kNN search

        :param proj: Projections of the dataset for this tree
        :type proj: np.ndarray

        :param indices: The indices of the projections for this tree
        :type indices: np.ndarray

        :param splits: (2^depth-1) array of splits corresponding to leaves
                       (tree, where immediate descendants follow parents;
                       index i's children are 2i+1 and 2i+2
        :type splits: np.ndarray

        :param split_index: The index of the current split element
        :type split_index: int

        :param level: Current level of tree
        :type level: int

        :return: Tree of splits and list of index arrays for each leaf
        :rtype: list[np.ndarray]
        """
        # If we're at the bottom, no split, just return the set
        if level == self._depth:
            return [indices]

        # Get the random projections for these indices at this level
        level_proj = proj[indices, level]
        n = indices.shape[0]

        # Split at the median if even, put median in upper half if not
        n_split = n // 2
        if n % 2 == 0:
            part_indices = np.argpartition(level_proj, (n_split - 1, n_split))
            split_val = level_proj[part_indices[n_split - 1]]
            split_val += level_proj[part_indices[n_split]]
            split_val /= 2.0
        else:
            part_indices = np.argpartition(level_proj, n_split)
            split_val = level_proj[part_indices[n_split]]

        splits[split_index] = split_val

        # part_indices is relative to this block of values, recover
        # main indices
        left_indices = indices[part_indices[:n_split]]
        right_indices = indices[part_indices[n_split:]]

        # Descend into each split and get sub-splits
        left_out = self._build_single_tree(
            proj, left_indices, splits, split_index=2 * split_index + 1,
            level=level + 1)
        right_out = self._build_single_tree(
            proj, right_indices, splits, split_index=2 * split_index + 2,
            level=level + 1)

        # Assemble index set
        return left_out + right_out

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
        super(MRPTNearestNeighborsIndex, self).build_index(descriptors)

        self._log.info("Building new MRPT index")

        self._log.debug("Storing descriptors")
        self._descr_cache = list(descriptors)
        if not self._descr_cache:
            raise ValueError("No data provided in given iterable.")
        # Cache descriptors if we have a path
        if self._descr_cache_filepath:
            self._log.debug("Caching descriptors: %s",
                            self._descr_cache_filepath)
            safe_create_dir(osp.dirname(self._descr_cache_filepath))
            with open(self._descr_cache_filepath, b"wb") as f:
                pickle.dump(self._descr_cache, f, -1)

        self._log.debug("Accumulating descriptor vectors into matrix")
        # XXX is an interval of 1.0 really a good idea?
        pts_array = elements_to_matrix(self._descr_cache, report_interval=1.0)

        self._log.debug('Building MRPT index')
        self._build_multiple_trees(pts_array)
        del pts_array

        self._log.debug("Caching index and parameters: %s, %s",
                        self._index_filepath, self._index_param_filepath)
        if self._index_filepath:
            self._log.debug("Caching index: %s", self._index_filepath)
            safe_create_dir(osp.dirname(self._index_filepath))
            with open(self._index_filepath, b"wb") as f:
                pickle.dump(self._trees, f, -1)

        if self._index_param_filepath:
            self._log.debug("Caching index params: %s",
                            self._index_param_filepath)
            params = {
                "num_trees": self._num_trees,
                "depth": self._depth,
                "required_votes": self._required_votes,
            }
            safe_create_dir(osp.dirname(self._index_param_filepath))
            with open(self._index_param_filepath, b"w") as f:
                pickle.dump(params, f, -1)

        self._pid = multiprocessing.current_process().pid

    def _load_mrpt_model(self):
        if not self._descr_cache and self._descr_cache_filepath:
            # Load descriptor cache
            # - is copied on fork, so only need to load here.
            self._log.debug("Loading cached descriptors")
            with open(self._descr_cache_filepath, b"rb") as f:
                self._descr_cache = pickle.load(f)

        if self._index_param_filepath:
            with open(self._index_param_filepath) as f:
                params = pickle.load(f)
            self._num_trees = params['num_trees']
            self._depth = params['depth']
            self._required_votes = params['required_votes']

        # Load the binary index
        if self._index_filepath:
            with open(self._index_filepath, b"rb") as f:
                self._trees = pickle.load(f)

        # Set current PID to the current
        self._pid = multiprocessing.current_process().pid

    def _restore_index(self):
        """
        If we think we're supposed to have an index, check the recorded PID with
        the current PID, reloading the index from cache if they differ.

        If there is a loaded index and we're on the same process that created it
        this does nothing.
        """
        if self._pid == multiprocessing.current_process().pid:
            return

        if bool(self._trees) and self._has_model_files():
            self._load_mrpt_model()

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        super(MRPTNearestNeighborsIndex, self).count()
        return len(self._descr_cache) if self._descr_cache else 0

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

        super(MRPTNearestNeighborsIndex, self).nn(d, n)

        def _query_single(tree):
            random_basis = tree['random_basis']
            depth = random_basis.shape[1]
            proj_query = d.vector() * random_basis
            splits = tree['splits']
            idx = 0
            for level in range(depth):
                split_point = splits[idx]
                # Look at the level'th coordinate of proj_query
                if proj_query[level] < split_point:
                    idx = 2 * idx + 1
                else:
                    idx = 2 * idx + 2

            # idx will be `2^depth - 1` greater than the position of the leaf
            # in the list
            idx -= ((1 << depth) - 1)
            return tree['leaves'][idx]

        def _exact_query(idcs):
            # Assemble the array to query from the descriptors that match
            pts_array = [self._descr_cache[idx].vector() for idx in idcs]
            pts_array = np.array(pts_array, dtype=pts_array[0].dtype)

            dists = ((pts_array - d.vector()) ** 2).sum(axis=1)

            if n >= dists.shape[0]:
                return idcs, dists

            near_indices = np.argpartition(dists, n - 1)[:n]
            return idcs[near_indices], dists[near_indices]

        # Use the sparse matrix trick to count the votes:
        # I.e. construct a sparse matrix with ones as data, one row, and the
        # indices as columns. After we add the votes for every tree, we extract
        # the indices with enough votes.
        votes = csr_matrix((1, len(self._descr_cache)), dtype=np.int32)

        for t in self._trees:
            leaf = _query_single(t)

            # Use the (data, indices, indptr) constructor. For one row, it's
            # simple
            votes += csr_matrix(
                (np.ones_like(leaf), leaf, [0, leaf.size]),
                shape=votes.shape, dtype=votes.dtype)

        # Votes will be in CSR format after addition. We get indices for free
        # after computing the comparison.
        sufficient_votes = votes.__ge__(self._required_votes)
        indices, distances = _exact_query(sufficient_votes.indices)
        order = distances.argsort()

        return ([self._descr_cache[indices[oidx]] for oidx in order],
                tuple(distances[oidx] for oidx in order))


NN_INDEX_CLASS = MRPTNearestNeighborsIndex
