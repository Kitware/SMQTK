"""
VP^s implementation of the VP Tree structure.
"""
import heapq
import random

import numpy

from smqtk.utils import INF
from smqtk.utils.vptree.vp import VpNode, _vp_select_vantage_random


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

    n.mu = numpy.median(p_dists)
    n.num_descendants = len(p_dists)

    # Items for left and right partitions.
    L = []
    R = []
    for i in item_list:
        if i.hist[-1] < n.mu:
            L.append(i)
        else:
            R.append(i)
    n.left = _vps_make_tree_inner(L, d)
    n.right = _vps_make_tree_inner(R, d)

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

    # TODO: Encode into a class.
    state = {
        'k': k,
        'q_distance': q_dist,
        # "Max" heap of neighbors. Python heapq always builds min-heaps, so we
        # store (-dist, node) elements. Most distance neighbor will always be
        # at top of heap due to distance negation.
        'neighbors': [],
        # Initial search radius. Whole tree considered, so tau is infinite to
        # start.
        'tau': INF,
    }

    # Start with the root node to search.
    _vps_knn_recursive_inner(state, root)

    # Unpack neighbor heap for return
    dists, neighbors = zip(*sorted(map(lambda dn: (-dn[0], dn[1]),
                                       state['neighbors'])))
    return neighbors, dists


def _vps_knn_recursive_inner(state, n):
    """
    Inner recursive search function for searching a VPS tree. Returns nothing
    but updates the state as appropriate.

    :param state: Search state dictionary.
    :type state: dict
    :param n: Tree node to consider.
    :type n: VpsNode
    """
    # Distance of query point from current node.
    d = state['q_distance'](n.p)

    # locally record reference to mutable heap container in the state.
    neighbors_heap = state['neighbors']

    if len(neighbors_heap) < state['k']:
        heapq.heappush(neighbors_heap, (-d, n.p))
    elif d <= state['tau']:
        # Add a k+1'th element to heap and remove the most distant candidate.
        heapq.heappushpop(neighbors_heap, (-d, n.p))
        # Set tau to the distance of the new most distance neighbor
        state['tau'] = -neighbors_heap[0][0]

    # Stop if n is a leaf
    if n.is_leaf():
        return

    # Descend into child nodes whose bounds potentially overlap the current tau
    # radius.
    if d < n.mu:
        # q inside mu radius, more likely intersects inner shell, but may
        # intersect outer shell.
        if _vps_check_in_bounds(n.left, d, state['tau']):
            _vps_knn_recursive_inner(state, n.left)
        if _vps_check_in_bounds(n.right, d, state['tau']):
            _vps_knn_recursive_inner(state, n.right)
    else:
        # q on/outside mu radius, more likely intersects outer shell, but may
        # intersect inner shell.
        if _vps_check_in_bounds(n.right, d, state['tau']):
            _vps_knn_recursive_inner(state, n.right)
        if _vps_check_in_bounds(n.left, d, state['tau']):
            _vps_knn_recursive_inner(state, n.left)


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
        # TODO: Check multiple children when there are more than two.
        if _vps_check_in_bounds(n.left, d, tau):
            to_search_push(n.left)
        if _vps_check_in_bounds(n.right, d, tau):
            to_search_push(n.right)

    # Need to negate the distances stored in list for return.
    for n in neighbors:
        n[0] = -n[0]
    dists, neighbors = zip(*sorted(neighbors))
    return neighbors, dists
