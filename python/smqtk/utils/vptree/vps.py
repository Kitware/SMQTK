"""
VP^s implementation of the VP Tree structure.
"""
import heapq
import random

import numpy

from smqtk.utils import INF
from smqtk.utils.vptree.vp import VpNode, _vp_select_vantage_random, \
    VpSearchState


__all__ = [
    'VpsNode',
    'vps_make_tree',
    'vps_knn_recursive',
    'vps_knn_iterative_heapq',
]


class VpsNode (VpNode):
    """
    Node structure used in a VPS tree.
    """

    def __init__(self):
        super(VpsNode, self).__init__()

        #
        # For VP-S implementation
        #
        # Tuple of tuples describing the bounds of this and child elements with
        # respect to ancestor nodes. Index 0 represents the root node, and
        # subsequent indices refer to subsequent ancestor nodes.
        # - bounds format: [(min, max), ...]
        #: :type: list[(float, float)]
        self.bounds = None

        # Tree level of this node could be saved during VPS construction by
        # observing how many ancestor distances a vantage point records upon
        # node construction.

    def to_arrays(
            self, value_array=None, mu_array=None, nd_array=None,
            bounds_array=None):
        """
        Store data from this VP tree as numpy arrays for use in serialization.

        The returned dictionary is suitable to be passed as the kwds argument
        to numpy.savez.

        :param value_array: An array to be filled with the node values of this
            and child nodes arranged as a binary tree.
        :type value_array: numpy.ndarray
        :param mu_array: An array to be filled with the mu values of this
            and child nodes arranged as a binary tree.
        :type mu_array: numpy.ndarray[float]
        :param nd_array: An array to be filled with the number of descendants
            in each branch of this node.
        :type nd_array: numpy.ndarray[numpy.ndarray[float]]
        :param bounds_array: An array to be filled with the bounds of each node
            in the tree.
        :type bounds_array: numpy.ndarray[object]
        :return: Dictionary with arrays as values and label strings as keys.
        :rtype: dict[str, numpy.ndarray]
        """
        max_children = getattr(self, "max_children", 2)
        if value_array is None:
            value_array = numpy.zeros(
                self.num_descendants + 1, dtype=numpy.dtype(type(self.p))
            )
        if mu_array is None:
            mu_array = numpy.zeros(self.num_descendants + 1, dtype='float')
        if nd_array is None:
            nd_array = numpy.zeros(
                (self.num_descendants + 1, max_children), dtype='int')
        if bounds_array is None:
            bounds_array = numpy.zeros(
                self.num_descendants + 1, dtype='object')

        if not len(value_array):
            return

        value_array[0] = self.p
        mu_array[0] = self.mu
        bounds_array[0] = self.bounds

        if self.children is not None:
            begin_index = 1
            end_index = 1
            for i, child in enumerate(self.children):
                if child is not None:
                    nd_array[0][i] = child.num_descendants + 1
                    begin_index = end_index
                    end_index += nd_array[0][i]
                    child.to_arrays(
                        value_array=value_array[begin_index:end_index],
                        mu_array=mu_array[begin_index:end_index],
                        nd_array=nd_array[begin_index:end_index],
                        bounds_array=bounds_array[begin_index:end_index],
                    )

        return {
            "value_array": value_array, "mu_array": mu_array,
            "nd_array": nd_array, "bounds_array": bounds_array
        }

    @classmethod
    def from_arrays(cls, value_array, mu_array, nd_array, bounds_array):
        """
        Construct VPS tree from arrays structured as binary trees.

        This method is useful for reconstructing VP trees from serialized data.

        :param value_array: An array with the node values of this and child
            nodes arranged as a binary tree.
        :type value_array: numpy.ndarray
        :param mu_array: An array with the mu values of this and child nodes
            arranged as a binary tree.
        :type mu_array: numpy.ndarray[float]
        :param nd_array: An array with the number of left and right descendants
            of each node in tree.
        :type nd_array: numpy.ndarray[float, float]
        :param bounds_array: An array with the bounds of each node in the tree.
        :type bounds_array: numpy.ndarray[object]
        :return: Reconstructed VPS tree.
        :rtype: VpsNode
        """
        if not len(value_array):
            return None
        new_node = cls()
        new_node.p = value_array[0]
        new_node.mu = mu_array[0]
        if numpy.isnan(new_node.mu):
            new_node.mu = None
        new_node.num_descendants = numpy.sum(nd_array[0])
        new_node.bounds = bounds_array[0]

        children = []
        begin_index = 1
        end_index = 1
        for nd in nd_array[0]:
            begin_index = end_index
            end_index += nd
            children.append(
                cls.from_arrays(
                    value_array[begin_index:end_index],
                    mu_array[begin_index:end_index],
                    nd_array[begin_index:end_index],
                    bounds_array[begin_index:end_index]
                )
            )
        if all(child is None for child in children):
            new_node.children = None
        else:
            new_node.children = tuple(children)
        return new_node


class VpsItem (object):
    """
    Item structure used in the construction of a VPS tree.
    """

    def __init__(self, id_):
        self.id = id_
        self.hist = []


def vps_make_tree(S, d, deduplicate=False, r_seed=None):
    """
    :param S: Metric space elements.
    :type S: collections.Iterable[object]

    :param d: Metric distance function: (a, b) -> [0, 1]
    :type d: (object, object) -> float

    :param deduplicate: Whether or not to deduplicate elements in ``S`` before
        creating the tree (don't create representative VpsItem for that
        element). Setting this to true requires that elements in ``S`` be
        hashable in order to use them in a set.
    :type deduplicate: bool

    :return: Root node of VPS tree.
    :rtype: VpsNode

    """
    if r_seed is not None:
        random.seed(r_seed)
    item_list = []
    # Set of unique elements in S in order to deduplicate input values.
    dedup_set = set()
    for s in S:
        if deduplicate:
            if s in dedup_set:
                continue
            else:
                dedup_set.add(s)
        i = VpsItem(s)
        item_list.append(i)
    return _vps_make_tree_inner(item_list, d)


def _vps_make_tree_inner(item_list, d):
    """
    :param item_list: "item" list.
    :type item_list: list[VpsItem]

    :param d: Metric distance function: (a, b) -> [0, 1]
    :type d: (object, object) -> float

    :return: Root node for this sub-tree.
    :rtype: VpsNode

    """
    if item_list is None or len(item_list) == 0:
        return None

    n = VpsNode()

    # Determine bounds of this node's children with respect to item distances
    # from the parent (current last distance in ``item.hist`` list).
    # - All item histories at this point should be of the same length, so we
    #   can pick the first one to determine level.
    n.bounds = []
    node_level = len(item_list[0].hist)
    for hist_i in range(node_level):
        dist_i = map(lambda i_: i_.hist[hist_i], item_list)
        n.bounds.append((min(dist_i), max(dist_i)))

    # One item in list, thus this is a leaf node.
    if len(item_list) == 1:
        i = item_list[0]
        n.p = i.id
        n.num_descendants = 0
        return n

    #: :type: VpsItem
    vp_i = _vp_select_vantage_random(item_list)
    del item_list[item_list.index(vp_i)]
    n.p = vp_i.id

    # Collect and retain child element distances to the current vantage point.
    p_dists = []  # parallel to current ``item_list``
    for i in item_list:
        di = d(n.p, i.id)
        i.hist.append(di)
        p_dists.append(di)

    n.num_descendants = len(p_dists)
    if n.num_descendants:
        n.mu = numpy.median(p_dists)

    # Items for left and right partitions.
    L = []
    R = []
    for i in item_list:
        if i.hist[-1] < n.mu:
            L.append(i)
        else:
            R.append(i)
    n.children = (_vps_make_tree_inner(L, d), _vps_make_tree_inner(R, d))

    return n


def _vps_check_in_bounds(n, dist, tau):
    # Check if the distance +- tau overlaps `n`s lowest-level bounds.
    return n and ((n.bounds[-1][0] - tau) <= dist <= (n.bounds[-1][1] + tau))


def vps_knn_recursive(q, k, root, dist_func):
    def q_dist(n):
        """
        :param n: Node value
        :type n: object
        :return: Metric distance to query value
        :rtype: float
        """
        return dist_func(q, n)

    state = VpSearchState(k, q_dist)

    # Start with the root node to search.
    _vps_knn_recursive_inner(state, root)

    # Unpack neighbor heap for return
    dists, neighbors = zip(*sorted(map(lambda dn: (-dn[0], dn[1]),
                                       state.neighbors)))
    return neighbors, dists


def _vps_knn_recursive_inner(state, n):
    """
    Inner recursive search function for searching a VPS tree. Returns nothing
    but updates the state as appropriate.

    :param state: Search state.
    :type state: VpSearchState
    :param n: Tree node to consider.
    :type n: VpsNode
    """
    # Distance of query point from current node.
    d = state.q_distance(n.p)

    # locally record reference to mutable heap container in the state.
    neighbors_heap = state.neighbors

    if len(neighbors_heap) < state.k:
        heapq.heappush(neighbors_heap, (-d, n.p))
    elif d <= state.tau:
        # Add a k+1'th element to heap and remove the most distant candidate.
        heapq.heappushpop(neighbors_heap, (-d, n.p))
        # Set tau to the distance of the new most distance neighbor
        state.tau = -neighbors_heap[0][0]

    # Stop if n is a leaf
    if n.is_leaf():
        return

    # Descend into child nodes whose bounds potentially overlap the current tau
    # radius.
    if d < n.mu:
        # q inside mu radius, more likely intersects inner shell, but may
        # intersect outer shell.
        if _vps_check_in_bounds(n.children[0], d, state.tau):
            _vps_knn_recursive_inner(state, n.children[0])
        if _vps_check_in_bounds(n.children[1], d, state.tau):
            _vps_knn_recursive_inner(state, n.children[1])
    else:
        # q on/outside mu radius, more likely intersects outer shell, but may
        # intersect inner shell.
        if _vps_check_in_bounds(n.children[1], d, state.tau):
            _vps_knn_recursive_inner(state, n.children[1])
        if _vps_check_in_bounds(n.children[0], d, state.tau):
            _vps_knn_recursive_inner(state, n.children[0])


def vps_knn_iterative_heapq(q, k, root, dist_func):
    """
    Get the ``k`` nearest neighbors to the query value ``q`` given the ``root``
    node of a VPS tree.

    ``root`` must be the result of calling ``vps_make_tree`` in order for nodes
    to have the correct properties for this flavor of search.

    This implementation uses a heap to rank the next branch to search down by
    distance from the query.

    :param q: Query value.
    :type q: object

    :param k: Number of nearest neighbors to return.
    :type k: int

    :param root: Root node of the VPS tree.
    :type root: VpsNode

    :param dist_func: Metric distance function that returns a floating point
        value in the [0, 1] range. This must be the same function used in the
        creation of the VPS tree.
    :type dist_func: (object, object) -> float

    :return: Two parallel list of the nearest neighbors values and their
        metric distances from the query, in order of increasing distance.
    :rtype: (list[object], list[float])

    """
    # max-Heap of near neighbors
    #   - python heapq does min-heap, so negating distances in heap
    neighbors = []
    tau = INF

    # Min-heap of vantage point candidates
    # - Heap elements are of the format: (float, VpsNode)
    to_search = []

    def to_search_push(n):
        """ Push a VpsNode onto the to-search heap. """
        heapq.heappush(to_search, (dist_func(q, n.p), n))

    to_search_push(root)

    while to_search:
        #: :type: (float, VpsNode)
        d, n = heapq.heappop(to_search)
        # ``d`` is the distance between ``q`` and ``n.p``

        if len(neighbors) < k:
            heapq.heappush(neighbors, [-d, n.p])
        elif d <= tau:
            # Add a k+1'th element to heap and remove the most distant
            # candidate.
            heapq.heappush(neighbors, [-d, n.p])
            heapq.heappop(neighbors)
            # Set tau to the distance of the new most distance neighbor
            tau = -neighbors[0][0]

        # shortcut if n is a leaf
        if n.is_leaf():
            continue

        # Add child nodes whose bounds potentially overlap the current tau
        # search radius
        for child in n.children:
            if _vps_check_in_bounds(child, d, tau):
                to_search_push(child)

    # Need to negate the distances stored in list for return.
    for n in neighbors:
        n[0] = -n[0]
    dists, neighbors = zip(*sorted(neighbors))
    return neighbors, dists
