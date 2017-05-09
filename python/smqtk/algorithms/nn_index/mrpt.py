# coding=utf-8
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

# noinspection PyPep8Naming
from six.moves import range, cPickle as pickle, zip

from os import path as osp

import numpy as np

from itertools import groupby

from smqtk.utils import plugin, merge_dict
from smqtk.algorithms.nn_index import NearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation import get_descriptor_index_impls
from smqtk.representation.descriptor_element import elements_to_matrix
from smqtk.utils.file_utils import safe_create_dir

CHUNK_SIZE = 5000


class MRPTNearestNeighborsIndex (NearestNeighborsIndex):
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
    tree. The neighbors are drawn from the set of points that are in the most
    leaves.

    The performance will depend on settings for the parameters:

    - If `depth` is too high, then the leaves will not have enough points
        to satisfy a query, and num_trees will need to be higher in order to
        compensate. If `depth` is too low, then performance may suffer because
        the leaves are large. If `N` is the size of the dataset, and `L =
        N/2^{depth}`, then leaves should be small enough that all
        `num_trees*L` descriptors that result from a query will fit easily in
        cache. Since query complexity is linear in `depth`, this parameter
        should be kept as low as possible.
    - The `num_trees` parameter will lower the variance of the results for
        higher values, but at the cost of using more memory on any particular
        query. As a rule of thumb, for a given value of `k`, num_trees should
        be about `3k/L`.
    """

    @classmethod
    def is_usable(cls):
        return True

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
        default = super(MRPTNearestNeighborsIndex, cls).get_default_config()

        di_default = plugin.make_config(get_descriptor_index_impls())
        default['descriptor_set'] = di_default

        return default

    def __init__(self, descriptor_set, index_filepath=None,
                 parameters_filepath=None, read_only=False,
                 # Parameters for building an index
                 num_trees=10, depth=1, random_seed=None,
                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                 use_multiprocessing=False):
        """
        Initialize MRPT index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param descriptor_set: Index in which DescriptorElements will be
            stored.
        :type descriptor_set: smqtk.representation.DescriptorIndex

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
        super(MRPTNearestNeighborsIndex, self).__init__()

        self._read_only = read_only
        self._use_multiprocessing = use_multiprocessing
        self._descriptor_set = descriptor_set
        self._pickle_protocol = pickle_protocol

        def normpath(p):
            return (p and osp.abspath(osp.expanduser(p))) or p

        self._index_filepath = normpath(index_filepath)
        self._index_param_filepath = normpath(parameters_filepath)
        # Now they're either None or an absolute path

        # parameters for building an index
        if depth < 1:
            raise ValueError("The depth may not be less than 1.")
        self._depth = depth
        if num_trees < 1:
            raise ValueError("The number of trees must be positive.")
        self._num_trees = num_trees

        # Set the list of trees to an empty list to have a sane value
        self._trees = []

        #: :type: None | int
        self._rand_seed = None
        if random_seed:
            self._rand_seed = int(random_seed)

        # Load the index/parameters if one exists
        if self._has_model_files():
            self._log.debug("Found existing model files. Loading.")
            self._load_mrpt_model()

    def get_config(self):
        return {
            "descriptor_set": plugin.to_plugin_config(self._descriptor_set),
            "index_filepath": self._index_filepath,
            "parameters_filepath": self._index_param_filepath,
            "read_only": self._read_only,
            "random_seed": self._rand_seed,
            "pickle_protocol": self._pickle_protocol,
            "use_multiprocessing": self._use_multiprocessing,
            "depth": self._depth,
            "num_trees": self._num_trees,
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
        :rtype: MRPTNearestNeighborsIndex

        """
        if merge_default:
            cfg = cls.get_default_config()
            merge_dict(cfg, config_dict)
        else:
            cfg = config_dict

        cfg['descriptor_set'] = \
            plugin.from_plugin_config(cfg['descriptor_set'],
                                      get_descriptor_index_impls())

        return super(MRPTNearestNeighborsIndex, cls).from_config(cfg, False)

    def _has_model_files(self):
        """
        check if configured model files are configured and exist
        """
        return (self._index_filepath and
                osp.isfile(self._index_filepath) and
                self._index_param_filepath and
                osp.isfile(self._index_param_filepath))

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
        if self._read_only:
            raise ReadOnlyError("Cannot modify container attributes due to "
                                "being in read-only mode.")

        super(MRPTNearestNeighborsIndex, self).build_index(descriptors)

        self._log.info("Building new MRPT index")

        self._log.debug("Clearing and adding new descriptor elements")
        self._descriptor_set.clear()
        self._descriptor_set.add_many_descriptors(descriptors)

        self._log.debug('Building MRPT index')
        self._build_multiple_trees()

        self._save_mrpt_model()

    def _build_multiple_trees(self, chunk_size=CHUNK_SIZE):
        """
        Build an MRPT structure
        """

        sample = self._descriptor_set.iterdescriptors().next()
        sample_v = sample.vector()
        n = self.count()
        d = sample_v.size
        leaf_size = n / (1 << self._depth)

        self._log.debug(
            "Building %d trees (T) of depth %d (l) from %g descriptors (N) "
            "of length %g",
            self._num_trees, self._depth, n, d)
        self._log.debug(
            "Leaf size             (L = N/2^l)  ~ %g/2^%d = %g",
            n, self._depth, leaf_size)
        self._log.debug(
            "UUIDs stored                (T*N)  = %g * %g = %g",
            self._num_trees, n, self._num_trees*n)
        self._log.debug(
            "Examined UUIDs              (T*L)  ~ %g * %g = %g",
            self._num_trees, leaf_size, self._num_trees*leaf_size)
        self._log.debug(
            "Examined/DB size  (T*L/N = T/2^l)  ~ %g/%g = %.3f",
            self._num_trees*leaf_size, n, self._num_trees*leaf_size/n)

        if (1 << self._depth) > n:
            self._log.warn(
                "There are insufficient elements (%d < 2^%d) to populate "
                "all the leaves of the tree. Consider lowering the depth "
                "parameter.", n, self._depth)

        self._log.debug("Projecting onto random bases")
        # Build all the random bases and the projections at the same time
        # (_num_trees * _depth shouldn't really be that high -- if it is,
        # you're a monster)
        random_bases = np.random.randn(self._num_trees, d, self._depth)
        projs = np.empty((n, self._num_trees, self._depth), dtype=np.float64)
        # Load the data in chunks (because n * d IS high)
        pts_array = np.empty((chunk_size, d), sample_v.dtype)
        # Enumerate the descriptors and div the index by the chunk size
        for k, g in groupby(enumerate(self._descriptor_set.iterdescriptors()),
                            lambda pair: pair[0] // chunk_size):
            # Items are still paired so extract the descriptors
            chunk = list(desc for i, desc in g)
            # Take care of dangling end piece
            k_beg = k * chunk_size
            k_end = min((k+1) * chunk_size, n)
            k_len = k_end - k_beg
            # Run the descriptors through elements_to_matrix
            elements_to_matrix(
                chunk, mat=pts_array, report_interval=1.0,
                use_multiprocessing=self._use_multiprocessing)
            # Insert into projection matrix
            projs[k_beg:k_end] = pts_array[:k_len].dot(random_bases)
        del pts_array

        self._log.debug("Constructing trees")
        desc_ids = list(self._descriptor_set.iterkeys())
        # Start with no trees
        self._trees = []
        for t in range(self._num_trees):
            # Array of splits is a packed tree
            splits = np.empty(((1 << self._depth) - 1,), np.float64)

            self._log.debug("Constructing tree #%d", t+1)

            # Build the tree & store it
            leaves = self._build_single_tree(projs[:, t], splits)
            leaves = [[desc_ids[idx] for idx in leaf]
                      for leaf in leaves]
            self._trees.append({
                'random_basis': (random_bases[t]),
                'splits': splits,
                'leaves': leaves
            })

    def _build_single_tree(self, proj, splits):
        """
        Build a single RP tree for fast kNN search

        :param proj: Projections of the dataset for this tree
        :type proj: np.ndarray (N, levels)

        :param splits: (2^depth-1) array of splits corresponding to leaves
                       (tree, where immediate descendants follow parents;
                       index i's children are 2i+1 and 2i+2
        :type splits: np.ndarray

        :return: Tree of splits and list of index arrays for each leaf
        :rtype: list[np.ndarray]
        """
        def _build_recursive(indices, level=0, split_index=0):
            """
            Descend recursively into tree to build it, setting splits and
            returning indices for leaves

            :param indices: The current set of indices before partitioning
            :param level: The level in the tree
            :param split_index: The index of the split to set

            :return: A list of arrays representing leaf membership
            :rtype: list[np.ndarray]
            """
            # If we're at the bottom, no split, just return the set
            if level == self._depth:
                return [indices]

            n = indices.size
            # If we literally don't have enough to populate the leaf, make it
            # empty
            if n < 1:
                return []

            # Get the random projections for these indices at this level
            # NB: Recall that the projection matrix has shape (levels, N)
            level_proj = proj[indices, level]

            # Split at the median if even, put median in upper half if not
            n_split = n // 2
            if n % 2 == 0:
                part_indices = np.argpartition(
                    level_proj, (n_split - 1, n_split))
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
            left_out = _build_recursive(left_indices, level=level + 1,
                                        split_index=2 * split_index + 1)
            right_out = _build_recursive(right_indices, level=level + 1,
                                         split_index=2 * split_index + 2)

            # Assemble index set
            left_out.extend(right_out)
            return left_out

        return _build_recursive(np.arange(proj.shape[0]))

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
                "read_only": self._read_only,
                "num_trees": self._num_trees,
                "depth": self._depth,
            }
            # noinspection PyTypeChecker
            with open(self._index_param_filepath, "w") as f:
                pickle.dump(params, f, self._pickle_protocol)

    def _load_mrpt_model(self):
        self._log.debug("Loading index and parameters: %s, %s",
                        self._index_filepath, self._index_param_filepath)
        if self._index_param_filepath:
            self._log.debug("Loading index params: %s",
                            self._index_param_filepath)
            with open(self._index_param_filepath) as f:
                params = pickle.load(f)
            self._read_only = params['read_only']
            self._num_trees = params['num_trees']
            self._depth = params['depth']

        # Load the index
        if self._index_filepath:
            self._log.debug("Loading index: %s", self._index_filepath)
            # noinspection PyTypeChecker
            with open(self._index_filepath, "rb") as f:
                self._trees = pickle.load(f)

    def count(self):
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        return len(self._descriptor_set)

    def nn(self, d, n=1):
        """
        Return the nearest `N` neighbors to the given descriptor element.

        :param d: Descriptor element to compute the neighbors of.
        :type d: smqtk.representation.DescriptorElement

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N DescriptorElement instances, and a tuple
            of the distance values to those neighbors.
        :rtype: (tuple[smqtk.representation.DescriptorElement], tuple[float])

        """
        super(MRPTNearestNeighborsIndex, self).nn(d, n)

        self._log.debug("Received query for %d nearest neighbors", n)

        depth, ntrees, db_size = self._depth, self._num_trees, self.count()
        leaf_size = db_size//(1 << depth)
        if leaf_size * ntrees < n:
            self._log.warning(
                "The number of descriptors in a leaf (%d) times the number "
                "of trees (%d) is less than the number of descriptors "
                "requested by the query (%d). The query result will be "
                "deficient.", leaf_size, ntrees, n)

        def _query_single(tree):
            # Search a single tree for the leaf that matches the query
            # NB: random_basis has shape (levels, N)
            random_basis = tree['random_basis']
            proj_query = d.vector().dot(random_basis)
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

        def _exact_query(_uuids):
            set_size = len(_uuids)
            self._log.debug("Exact query requested with %d descriptors",
                            set_size)

            # Assemble the array to query from the descriptors that match
            d_v = d.vector()
            pts_array = np.empty((set_size, d_v.size), dtype=d_v.dtype)
            descriptors = self._descriptor_set.get_many_descriptors(_uuids)
            for i, desc in enumerate(descriptors):
                pts_array[i, :] = desc.vector()

            dists = ((pts_array - d_v) ** 2).sum(axis=1)

            if n > dists.shape[0]:
                self._log.warning(
                    "There were fewer descriptors (%d) in the set than "
                    "requested in the query (%d). Returning entire set.",
                    dists.shape[0], n)
            if n >= dists.shape[0]:
                return _uuids, dists

            near_indices = np.argpartition(dists, n - 1)[:n]
            return ([_uuids[idx] for idx in near_indices],
                    dists[near_indices])

        # Take union of all tree hits
        tree_hits = set()
        for t in self._trees:
            tree_hits.update(_query_single(t))

        hit_union = len(tree_hits)
        self._log.debug(
            "Query (k): %g, Hit union (h): %g, DB (N): %g, "
            "Leaf size (L = N/2^l): %g, Examined (T*L): %g",
            n, hit_union, db_size, leaf_size, leaf_size * ntrees)
        self._log.debug("k/L     = %.3f", n / leaf_size)
        self._log.debug("h/N     = %.3f", hit_union / db_size)
        self._log.debug("h/L     = %.3f", hit_union / leaf_size)
        self._log.debug("h/(T*L) = %.3f", hit_union / (leaf_size * ntrees))

        uuids, distances = _exact_query(list(tree_hits))
        order = distances.argsort()
        uuids, distances = zip(
            *((uuids[oidx], distances[oidx]) for oidx in order))

        self._log.debug("Returning query result of size %g", len(uuids))

        return (tuple(self._descriptor_set.get_many_descriptors(uuids)),
                tuple(distances))


NN_INDEX_CLASS = MRPTNearestNeighborsIndex
