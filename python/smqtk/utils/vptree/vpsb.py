"""
VP^sb implementation.

This implementation adds a branching factor to tree construction and search.
This is different than the original author's description in that each node has
multiple children instead of only the leaf nodes having multiple data elements.
"""
import heapq
import itertools
import random

import numpy
import scipy.stats
from six.moves import range

from smqtk.utils import INF
from smqtk.utils.vptree.vp import _vp_select_vantage_random
from smqtk.utils.vptree.vps import VpsNode, VpsItem, _vps_check_in_bounds


__all__ = [
    'VpsbNode',
    'vpsb_make_tree',
    'vpsb_knn_recursive',
]


class VpsbNode (VpsNode):
    """
    A VPSB node differs from previous node classes in that ``mu`` is a tuple of
    distances corresponding to the number of children that node has
    """

    def __init__(self):
        super(VpsbNode, self).__init__()

        # Maximum number of children this node may have.
        #: :type: None | int
        self.max_children = None

        # Tuple of child nodes to this node.
        #: :type: None | tuple
        self.children = None

    def is_leaf(self):
        """
        :return: If this node has no children.
        :rtype: bool
        """
        return (self.children is None) or (len(self.children) == 0)

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
        children = getattr(self, "children", (self.left, self.right))
        if value_array is None:
            value_array = numpy.zeros(
                self.num_descendants + 1, dtype=numpy.dtype(type(self.p))
            )
        if mu_array is None:
            mu_array = numpy.zeros(self.num_descendants + 1, dtype='object')
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

        if children is not None:
            begin_index = 1
            end_index = 1
            for i, child in enumerate(children):
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
    def from_arrays(
            cls, value_array, mu_array, nd_array, bounds_array,
            max_children=None):
        """
        Construct VPSB tree from arrays structured as binary trees.

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
        :return: Reconstructed VPSB tree.
        :rtype: VpsNode
        """
        if not len(value_array):
            return None
        new_node = cls()
        new_node.p = value_array[0]
        new_node.mu = mu_array[0]
        new_node.num_descendants = numpy.sum(nd_array[0])
        new_node.bounds = bounds_array[0]

        children = []
        begin_index = 1
        end_index = 1
        if max_children is None:
            max_children = nd_array.shape[1]
        new_node.max_children = max_children
        for nd in nd_array[0]:
            begin_index = end_index
            end_index += nd
            children.append(
                cls.from_arrays(
                    value_array[begin_index:end_index],
                    mu_array[begin_index:end_index],
                    nd_array[begin_index:end_index],
                    bounds_array[begin_index:end_index],
                    max_children=max_children
                )
            )
        if all(child is None for child in children):
            new_node.children = None
        else:
            new_node.children = tuple(children)
        return new_node


def vpsb_make_tree(S, d, branching_factor=2, deduplicate=False, r_seed=None,
                   mu_selection='partition'):
    """
    Create a VP^SB tree structure.

    :param S: Metric space elements.
    :type S: collections.Iterable[object]

    :param d: Metric distance function: (a, b) -> [0, 1]
    :type d: (object, object) -> float

    :param branching_factor: Maximum number of children per node.
    :type branching_factor: int

    :param deduplicate: Whether or not to deduplicate elements in ``S`` before
        creating the tree (don't create representative VpsItem for that
        element). Setting this to true requires that elements in ``S`` be
        hashable in order to use them in a set.
    :type deduplicate: bool

    :param r_seed: Random number generator seed value.
    :type r_seed: None | int

    :param mu_selection: Mu value selection method. This must either be
        'partition' or 'quantile'.
        TODO: Explain the difference in methods.
    :type mu_selection: str

    :return: Root node of VPSB tree.
    :rtype: VpsbNode

    """
    if r_seed is not None:
        random.seed(r_seed)
    if mu_selection not in {'partition', 'quantile'}:
        raise ValueError("Unknown mu selection method: %s" % mu_selection)
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
    del dedup_set
    return _vpsb_make_tree_inner(item_list, d, branching_factor, mu_selection)


def _vpsb_make_tree_inner(item_list, d, branching_factor, mu_selection):
    """
    :param item_list: "item" list.
    :type item_list: list[VpsItem]

    :param d: Metric distance function: (a, b) -> [0, 1]
    :type d: (object, object) -> float

    :param branching_factor: Max number of children for a node.
    :type branching_factor: int

    :param mu_selection: MU selection method name.
    :type mu_selection: str

    :return: Root node for this sub-tree.
    :rtype: VpsbNode

    """
    if item_list is None or len(item_list) == 0:
        return None

    n = VpsbNode()
    n.max_children = branching_factor

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
        item = item_list[0]
        n.p = item.id
        n.num_descendants = 0
        return n

    #: :type: VpsItem
    vp_i = _vp_select_vantage_random(item_list)
    del item_list[item_list.index(vp_i)]
    n.p = vp_i.id
    n.num_descendants = len(item_list)

    # Collect and retain child element distances to the current vantage point.
    p_dists = []  # parallel to current ``item_list``
    for item in item_list:
        di = d(n.p, item.id)
        item.hist.append(di)
        p_dists.append(di)

    # Select mu values separating child spaces.
    children = min(branching_factor, len(item_list))

    if mu_selection == 'partition':
        # Selection (multi-median) method
        # - quickly selects indices of kth smallest elements.
        # - does not interpolate when median index is not integral, casts to
        #   int.
        interval = len(p_dists) / float(children)
        mu_indices = map(lambda i: int(interval * i), range(1, children))
        # Use a quick selection algorithm to find multiple "medians" in the
        # array of distances without sorting the whole list. ``n.mu`` values
        # will be in ascending order after ``numpy.partition``.
        # TODO: Check proportion of indices to be selected to the size of the
        #       array, simply sorting the array after proportion reaches a
        #       threshold as selecting a large portion of the array is slower
        #       than just sorting it.
        if not mu_indices:
            n.mu = []
        else:
            n.mu = numpy.partition(p_dists, mu_indices)[mu_indices]
    elif mu_selection == 'quantile':
        # Quantile method -- uneven spaced bins, even bin counts.
        # - basically more accurate version of selection method (getting
        #   similar results experimentally)
        # - quantile probability array is evenly spaced and given +1 in order
        #   to make the appropriate number of bins.
        quantile_probs = numpy.linspace(0, 1, children + 1)
        p_dist_quantiles = scipy.stats.mstats.mquantiles(
            p_dists, quantile_probs)
        # Mu values are the quantile values minus the min/max bounds.
        n.mu = p_dist_quantiles[1:-1]
    else:
        raise ValueError("Invalid mu selection method: %s" % mu_selection)

    # Sift items into bins based on mu values. Bins are in congruent order to
    # the mu value list, which is in ascending order (smallest to largest),
    # i.e. bin[0] are items closest to the vantage point and bin[-1] are items
    # farthest away.
    item_bins = tuple([] for _ in range(children))
    dist_bins = [-INF] + list(n.mu) + [INF]
    for item, bin_i in itertools.izip(item_list,
                                      numpy.digitize(p_dists, dist_bins)):
        # need the -1 because digitize's return is 1-index.
        item_bins[bin_i - 1].append(item)

    # Recurse for each child bin in order.
    n.children = map(lambda b: _vpsb_make_tree_inner(b, d, branching_factor,
                                                     mu_selection),
                     item_bins)

    return n


def vpsb_knn_recursive(q, k, root, dist_func):
    """
    K Nearest neighbor function for VP^sb tree.

    :param q: Query value.
    :type q: object

    :param k: Number of neighbors to find.
    :type k: int

    :param root: Tree root node.
    :type root: VpsbNode

    :param dist_func: Distance metric function: (a, b) -> [0, 1]. Should be the
        same function used when building the tree.
    :type dist_func: (object, object) -> float

    :return: Parallel lists of neighbor values and their distances from the
        query value.
    :rtype: object

    """
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
    _vpsb_knn_recursive_inner(state, root)

    # Unpack neighbor heap for return
    dists, neighbors = zip(*sorted(map(lambda dn: (-dn[0], dn[1]),
                                       state['neighbors'])))
    return neighbors, dists


def _vpsb_child_order_key(child, dist):
    """
    Generate the key value for a child node for potential recursive search
    order.
    """
    if child is None:
        return INF, INF
    else:
        # distance bounds from immediate parent node.
        c_bounds = child.bounds[-1]
        return (
            # If we're concretely in the bounds of the current child node
            # True (1) should be less than failure value, because sorting.
            int(c_bounds[0] <= dist <= c_bounds[1]) or 2,
            min(abs(dist - c_bounds[0]), abs(dist - c_bounds[1])),
        )


def _vpsb_knn_recursive_inner(state, n):
    """
    Inner recursive search function for searching a VP^sb tree. Returns nothing
    but updates the state as appropriate.

    :param state: Search state dictionary.
    :type state: dict

    :param n: Tree node to consider.
    :type n: VpsbNode

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

    # Order child nodes based on how close the query point is from that node's
    # bounds range. This is equivalent to checking on what side of the mu
    # boundary the query distance is.
    # - child node should never be None-valued since only actual children are
    #   stored during build.
    # map(lambda c: _vpsb_check_bounds(c, d, state[]), n.children)
    ordered_children = sorted(n.children,
                              key=lambda c: _vpsb_child_order_key(c, d))
    for node in ordered_children:
        if _vps_check_in_bounds(node, d, state['tau']):
            _vpsb_knn_recursive_inner(state, node)
