# coding=utf-8
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from os import path as osp

from six.moves import range, cPickle as pickle

# import logging

import numpy as np
from scipy.sparse import csr_matrix, random
from scipy import stats

from smqtk.utils import SmqtkObject
from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils.file_utils import safe_create_dir


class DescriptorCache(SmqtkObject):
    def __init__(self):
        self._descr_cache = None

    def init_descr_cache(self, descriptors, descr_cache_filepath,
                         pickle_protocol=pickle.HIGHEST_PROTOCOL):
        self._descr_cache = list(descriptors)
        if not self._descr_cache:
            raise ValueError("No data provided in given iterable.")

        # Cache descriptors if we have a path
        if descr_cache_filepath:
            self._log.debug("Caching descriptors: %s", descr_cache_filepath)
            safe_create_dir(osp.dirname(descr_cache_filepath))
            # noinspection PyTypeChecker
            with open(descr_cache_filepath, "wb") as f:
                pickle.dump(self._descr_cache, f, pickle_protocol)

    def load_descr_cache(self, descr_cache_filepath):
        if not self._descr_cache and descr_cache_filepath:
            # Load descriptor cache
            # - is copied on fork, so only need to load here.
            self._log.debug("Loading cached descriptors")
            # noinspection PyTypeChecker
            with open(descr_cache_filepath, "rb") as f:
                self._descr_cache = pickle.load(f)


class MRPTNearestNeighborsIndex (NearestNeighborsIndex, DescriptorCache):
    """
    Nearest Neighbors index that uses the MRPT algorithm of [Hyvönen et
    al](https://arxiv.org/abs/1509.06957).

    Multiple Random Projection Trees (MRPT) combines multiple shallow binary
    trees of a set depth to quickly search for near neighbors. Each tree has a
    separate set of random projections used to divide the dataset into a tree
    structure. This algorithm differs from most RP tree implementations in
    that all separators at a particular level in the tree share the same
    projection, to save on space. Every branch partitions a set of points into
    two equal portions relative to the corresponding random projection.

    On query, the leaf corresponding to the query vector is found in each
    tree. The neighbors are drawn from the set of points that are in at least
    a certain number of leaves (`required_votes`).

    The performance will depend on settings for the parameters:

    - Set `required_votes` to a small integer greater than 1. If it is too
        high, then a query may not return enough points. Higher values can
        lower the variance of results, but `required_votes` should always be
        much smaller than the number of trees.
    - If `depth` is too high, then the leaves will not have enough points
        to satisfy a query, and many trees will be required to compensate. If
        `depth` is too low, then performance may suffer because the leaves are
        large. If `N` is the size of the dataset, and `L = N/2^{depth}`, then
        leaves should be small enough that all `num_trees*L` descriptors that
        result from a query will fit easily in cache. Since query complexity
        is linear in `depth`, this parameter should be kept as low as
        possible.
    - The `num_trees` parameter will lower the variance of the results for
        higher values, but at the cost of using more memory on any particular
        query. As a rule of thumb, for a given value of `k`, num_trees should
        be about `3k/L`.
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, index_filepath=None, parameters_filepath=None,
                 descriptor_cache_filepath=None,
                 # Parameters for building an index
                 num_trees=10, depth=1, required_votes=3, random_seed=None,
                 pickle_protocol=pickle.HIGHEST_PROTOCOL):
        """
        Initialize MRPT index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param pickle_protocol: The protocol version to be used by the pickle
            module to serialize class information
        :type pickle_protocol: int

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
        :type num_trees: int

        :param depth: The depth of the trees
        :type depth: int

        :param required_votes: The number of votes required for a nearest
            neighbor
        :type required_votes: int

        :param random_seed: Integer to use as the random number generator seed.
        :type random_seed: int

        """
        super(MRPTNearestNeighborsIndex, self).__init__()

        self._pickle_protocol = pickle_protocol

        def normpath(p):
            return (p and osp.abspath(osp.expanduser(p))) or p

        self._index_filepath = normpath(index_filepath)
        self._index_param_filepath = normpath(parameters_filepath)
        self._descr_cache_filepath = normpath(descriptor_cache_filepath)
        # Now they're either None or an absolute path

        # parameters for building an index
        if depth < 0:
            # TODO Handle the zero-depth case, which is just exact NN
            raise ValueError("The depth may not be negative.")
        self._depth = depth
        if num_trees < 1:
            raise ValueError("The number of trees must be positive.")
        self._num_trees = num_trees
        if required_votes < 1:
            raise ValueError("The number of required votes must be positive")
        self._required_votes = required_votes

        # Set the list of trees to an empty list to have a sane value
        self._trees = []

        # In-order cache of descriptors we're indexing over.
        # - index will spit out indices to list
        #: :type: list[smqtk.representation.DescriptorElement] | None

        #: :type: None | int
        self._rand_seed = None
        if random_seed:
            self._rand_seed = int(random_seed)

        # Load the index/parameters if one exists
        if self._has_model_files():
            self._log.info("Found existing model files. Loading.")
            self._load_mrpt_model()

        if (self._descr_cache_filepath and
                osp.isfile(self._descr_cache_filepath)):
            self.load_descr_cache(self._descr_cache_filepath)

    def get_config(self):
        return {
            "index_filepath": self._index_filepath,
            "parameters_filepath": self._index_param_filepath,
            "descriptor_cache_filepath": self._descr_cache_filepath,
            "random_seed": self._rand_seed,
            "pickle_protocol": self._pickle_protocol,
            "depth": self._depth,
            "num_trees": self._num_trees,
            "required_votes": self._required_votes,
        }

    def _has_model_files(self):
        """
        check if configured model files are configured and exist
        """
        return (self._index_filepath and osp.isfile(self._index_filepath) and
                self._index_param_filepath and osp.isfile(self._index_param_filepath))

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
        self.init_descr_cache(descriptors, self._descr_cache_filepath,
                              pickle_protocol=self._pickle_protocol)

        self._log.debug("Accumulating descriptor vectors into matrix")
        pts_array = elements_to_matrix(self._descr_cache, report_interval=1.0)

        self._log.debug('Building MRPT index')
        self._build_multiple_trees(pts_array)
        del pts_array

        self._save_mrpt_model()

    def _build_multiple_trees(self, pts):
        """
        Build an MRPT structure for data pts
        :param pts: The data. Each row is a datum.
        :type pts: np.ndarray
        """
        n, d = pts.shape

        # Do transposition once
        _ptsT = pts.T

        # Get the Normal distribution RNG
        rvs = stats.norm().rvs
        # Start with no trees
        self._trees = []
        # 1/sqrt(depth) considered optimal for random projections
        density = 1 / np.sqrt(self._depth)
        for _ in range(self._num_trees):
            # Each tree has a basis of sparse random projections
            # NB: this matrix is constructed so that we can do left
            # multiplication rather than right multiplication -- otherwise a
            # transpose of the input matrix is incurred on every iteration
            random_basis = random(
                self._depth, d, density=density, format="csr",
                dtype=np.float64, random_state=self._rand_seed, data_rvs=rvs)
            # Array of splits is a packed tree
            splits = np.empty(((1 << self._depth) - 1,), np.float64)

            # Build the tree & store it
            leaves = self._build_single_tree(
                random_basis * _ptsT, np.arange(n), splits)
            self._trees.append({
                'random_basis': random_basis,
                'splits': splits,
                'leaves': leaves
            })

    def _build_single_tree(self, proj, indices, splits, split_index=0, level=0):
        """
        Build a single RP tree for fast kNN search

        :param proj: Projections of the dataset for this tree
        :type proj: np.ndarray (levels, N)

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
        # NB: Recall that the projection matrix has shape (levels, N)
        level_proj = proj[level, indices]
        n = indices.size

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

    def _save_mrpt_model(self):
        self._log.debug("Caching index and parameters: %s, %s",
                        self._index_filepath, self._index_param_filepath)
        if self._index_filepath:
            self._log.debug("Caching index: %s", self._index_filepath)
            safe_create_dir(osp.dirname(self._index_filepath))
            # noinspection PyTypeChecker
            with open(self._index_filepath, "wb") as f:
                pickle.dump(self._trees, f, self._pickle_protocol)
        if self._index_param_filepath:
            self._log.debug("Caching index params: %s",
                            self._index_param_filepath)
            safe_create_dir(osp.dirname(self._index_param_filepath))
            params = {
                "num_trees": self._num_trees,
                "depth": self._depth,
                "required_votes": self._required_votes,
            }
            # noinspection PyTypeChecker
            with open(self._index_param_filepath, "w") as f:
                pickle.dump(params, f, self._pickle_protocol)

    def _load_mrpt_model(self):
        if self._index_param_filepath:
            # noinspection PyTypeChecker
            with open(self._index_param_filepath) as f:
                params = pickle.load(f)
            self._num_trees = params['num_trees']
            self._depth = params['depth']
            self._required_votes = params['required_votes']

        # Load the index
        if self._index_filepath:
            # noinspection PyTypeChecker
            with open(self._index_filepath, "rb") as f:
                self._trees = pickle.load(f)

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
        super(MRPTNearestNeighborsIndex, self).nn(d, n)

        def _query_single(tree):
            # Search a single tree for the leaf that matches the query
            # NB: random_basis has shape (levels, N)
            random_basis = tree['random_basis']
            depth = random_basis.shape[0]
            proj_query = random_basis * d.vector()
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
        sufficient_votes = (votes >= self._required_votes)
        indices, distances = _exact_query(sufficient_votes.indices)
        order = distances.argsort()

        return ([self._descr_cache[indices[oidx]] for oidx in order],
                tuple(distances[oidx] for oidx in order))


NN_INDEX_CLASS = MRPTNearestNeighborsIndex
