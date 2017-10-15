"""
VP Tree implementation functions
"""
import collections
import heapq
import random

import numpy
import six

from smqtk.utils import INF


__all__ = [
    'VpNode',
    'vp_make_tree',
    'vp_knn_recursive',
    'vp_knn_iterative',
    'vp_knn_iterative_heapq',
]


class VpNode (object):

    def __init__(self):
        # node value
        #: :type: object
        self.p = None
        # Median distance to children
        #: :type: float
        self.mu = None
        # Number of nodes that can be found as descendants underneath this
        # node, including immediate children.
        #: :type: int | long
        self.num_descendants = None

        # Tuple of child nodes to this node
        #: :type: None | tuple
        self.children = None

    def __repr__(self):
        return "%s{p: %s, mu: %s, descendants: %d}" \
               % (self.__class__.__name__, self.p, self.mu,
                  self.num_descendants)

    def to_arrays(self, value_array=None, mu_array=None, nd_array=None):
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
        :param nd_array: An array to be filled with the number of left and
            right descendants of each node in tree.
        :type nd_array: numpy.ndarray[float, float]
        :return: Dictionary with arrays as values and label strings as keys.
        :rtype: dict[str, numpy.ndarray]
        """
        if value_array is None:
            if type(self.p) is long:
                dtype = numpy.dtype('object')
            else:
                dtype = numpy.dtype(type(self.p))
            value_array = numpy.zeros(
                self.num_descendants + 1, dtype=dtype
            )
        if mu_array is None:
            mu_array = numpy.zeros(self.num_descendants + 1, dtype='float')
        if nd_array is None:
            nd_array = numpy.zeros((self.num_descendants + 1, 2), dtype='int')

        if not len(value_array):
            return

        value_array[0] = self.p
        mu_array[0] = self.mu

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
                    )
        return {
            "value_array": value_array, "mu_array": mu_array,
            "nd_array": nd_array
        }

    @classmethod
    def from_arrays(cls, value_array, mu_array, nd_array):
        """
        Construct VP tree from arrays structured as binary trees.

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
        :return: Reconstructed VP tree.
        :rtype: VpNode
        """
        if not len(value_array):
            return None
        new_node = cls()
        new_node.p = value_array[0]
        new_node.mu = mu_array[0]
        new_node.num_descendants = numpy.sum(nd_array[0])
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
                )
            )
        if all(child is None for child in children):
            new_node.children = None
        else:
            new_node.children = tuple(children)
        return new_node

    def is_leaf(self):
        """
        :return: If this node is a leaf in the tree.
        :rtype: bool
        """
        return self.children is None


def vp_make_tree(S, d, r_seed=None):
    """
    Create VP Tree, returning the root node.

    Elements of S must be hashable for use with element de-duplication.

    Side effect: Duplicate values in S are ignored such that only unique values
        in S are stored.

    :param S: metric space elements
    :param d: distance function between two elements of S. This should yield
        distances in the [0,1] range.
    :param r_seed: random seed int

    :return: Root node of the tree
    :rtype: VpNode

    """
    if r_seed is not None:
        random.seed(r_seed)
    # TODO: Convert S to a numpy object array to leverage array view slicing.
    return _vp_make_tree_inner(S, d)


def _vp_make_tree_inner(S, d):
    """
    Side effect: Duplicate values in S are ignored such that only unique values
        in S are stored.

    :param S: metric space elements
    :param d: distance function between two elements of S. This should yield
        distances in the [0,1] range.

    :return: Root node of the tree
    :rtype: VpNode

    """
    if S is None or len(S) == 0:
        return None

    # TODO: make local sequence of S in case S is an iterable.

    n = VpNode()

    # Only one item left, this is a leaf node.
    if len(S) == 1:
        n.p = S[0]
        n.mu = d(n.p, n.p)  # or 0 if assuming 0 distance to self.
        n.num_descendants = 0
        return n

    n.p = _vp_select_vantage_random(S)
    # n.p = vp_select_vantage_probabilistic(S, d, r)

    S_d = dict((s_, d(n.p, s_)) for s_ in S)
    del S_d[n.p]  # remove vantage point from children to consider.
    n.mu = numpy.median(list(six.itervalues(S_d)))
    n.num_descendants = len(S_d)

    L = []
    R = []
    for s, dist in six.iteritems(S_d):
        # The vantage point was removed from this mapping earlier, so do not
        # have to worry about that.
        if dist < n.mu:
            L.append(s)
        else:
            R.append(s)
    n.children = (_vp_make_tree_inner(L, d), _vp_make_tree_inner(R, d))

    return n


def _vp_select_vantage_random(S):
    """
    Select random point from ``S`` to be the vantage point.
    :param S: Set of elements.
    :type S: collections.Sequence

    :return: Randomly selected element within ``S``.
    :rtype: object

    """
    return S[random.randint(0, len(S) - 1)]


def vp_select_vantage_probabilistic(S, d, r):
    """
    Select a vantage point element.

    :param S: metric space elements
    :type S: collection.Sequence[object]

    :param d: distance function between two elements of S. This should yield
        distances in the [0,1] range.
    :type d: (object, object) -> float

    :param r: random subsample percentage in the [0, 1] range.
    :type r: float

    :return: Vantage point value.
    :rtype: object

    """
    # noinspection PyDefaultArgument
    def get_d(a, b, cache={}):
        # faster than: i, j = (min(a, b), max(a, b))
        i, j = ((a < b) and (a, b)) or (b, a)
        if not (i in cache and j in cache[i]):
            cache.setdefault(i, {})[j] = d(i, j)
        return cache[i][j]

    # Pick a sample of S as possible vantage points.
    P = random.sample(S, int(max(round(len(S) * r), 1)))
    best_spread = None
    best_p = None
    for p in P:
        # Pick a sample of S to test vantage point position.
        D = random.sample(S, int(max(round(len(S) * r), 1)))
        pD_dists = numpy.array([get_d(p, di) for di in D])
        mu = numpy.mean(pD_dists)
        spread = numpy.std(pD_dists - mu)
        if best_spread is None or spread > best_spread:
            best_spread = spread
            best_p = p
    return best_p


def vp_knn_recursive(q, k, root, d_func):
    """
    Fine ``k`` nearest neighbors in tree with starting at ``root``.

    :param q: Query value.
    :type q: object

    :param k: Number of neighbors to return.
    :type k: int

    :param root: Root node of VP tree. Must have been constructed with
        ``vp_make_tree`` function.
    :type root: VpNode

    :param d_func: Distance metric function, which should return a value in the
        [0, 1] range.
    :type d_func: (object, object) -> float

    :return: Parallel lists of near values and distances of those values from
        the query node.
    :rtype: (tuple[object], tuple[float])

    """
    def q_dist(n):
        """
        :param n: Node value
        :type n: object
        :return: Metric distance to query value
        :rtype: float
        """
        return d_func(q, n)

    # TODO: Encode into a class.
    state = {
        'k': k,
        'q_distance': q_dist,
        # "Max"" heap of neighbors. Python heapq always builds min-heaps, so we
        # store (-dist, node) elements. Most distance neighbor will always be
        # at top of heap due to distance negation.
        'neighbors': [],
        # Initial search radius. Whole tree considered, so tau is infinite to
        # start.
        'tau': INF,
    }

    # Start with the root node to search.
    _vp_knn_recursive_inner(state, root)

    # Unpack neighbor heap for return
    dists, neighbors = zip(*sorted(map(lambda dn: (-dn[0], dn[1]),
                                       state['neighbors'])))
    return neighbors, dists


def _vp_knn_recursive_inner(state, n):
    """
    Inner recursive search function. Returns nothing but updates the state as
    appropriate.

    :param state: Search state dictionary
    :type state: dict
    :param n: Tree node to consider.
    :type n: VpNode

    """
    # Distance value from the current node's value to the query value.
    d = state['q_distance'](n.p)

    # locally record reference to mutable heap container in the state.
    neighbors_heap = state['neighbors']

    if len(state['neighbors']) < state['k']:
        heapq.heappush(neighbors_heap, (-d, n.p))
    elif d <= state['tau']:
        # Add current node as the k+1'th element to heap and remove the then
        # most distant candidate.
        heapq.heappushpop(neighbors_heap, (-d, n.p))
        # Update tau to the distance of the new most distance neighbor, which
        # should be the top of the heap.
        state['tau'] = -neighbors_heap[0][0]

    # Stop if n is a leaf
    if n.is_leaf():
        return

    if d < n.mu:
        # Should at least check left child, optionally right if distance
        # radius overlaps right region.
        if n.children[0]:  # d < n.mu + tau
            _vp_knn_recursive_inner(state, n.children[0])
        if n.children[1] and d >= n.mu - state['tau']:
            _vp_knn_recursive_inner(state, n.children[1])
    else:
        # Should at least check right child, optionally left if distance
        # radius overlaps left region.
        if n.children[1]:  # d >= n.mu - tau
            _vp_knn_recursive_inner(state, n.children[1])
        if n.children[0] and d < n.mu + state['tau']:
            _vp_knn_recursive_inner(state, n.children[0])


def vp_knn_iterative(q, k, root, distance):
    """
    Nearest neighbors only search from a query value ``q``.
    """
    # max-Heap of near neighbors
    #   - python heapq does min-heap, so negating distances in heap
    neighbors = []
    tau = float('inf')
    #: :type: collections.deque[VpNode]
    to_search = collections.deque()
    to_search.append(root)

    while to_search:
        n = to_search.popleft()
        d = distance(q, n.p)

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

        if d < n.mu:
            # Should at least check left child, optionally right if distance
            # radius overlaps right region.
            if n.children[0]:  # d < n.mu + tau
                to_search.append(n.children[0])
            if n.children[1] and d >= n.mu - tau:
                to_search.append(n.children[1])
        else:
            # Should at least check right child, optionally left if distance
            # radius overlaps left region.
            if n.children[1]:  # d >= n.mu - tau
                to_search.append(n.children[1])
            if n.children[0] and d < n.mu + tau:
                to_search.append(n.children[0])

    # Need to negate the negated distances for return
    for n in neighbors:
        n[0] = -n[0]
    dists, neighbors = zip(*sorted(neighbors))
    return neighbors, dists


def vp_knn_iterative_heapq(q, k, root, distance):
    """
    Nearest neighbors only search from a query value ``q``.

    This is the slowest of the three versions due to the overhead of the
    ``to_search`` also being a heap.
    """
    # max-Heap of near neighbors
    #   - python heapq does min-heap, so negating distances in heap
    neighbors = []
    tau = INF

    # Min-heap of vantage point candidates
    to_search = []

    def to_search_push(n_):
        heapq.heappush(to_search, (distance(q, n_.p), n_))

    to_search_push(root)

    while to_search:
        d, n = heapq.heappop(to_search)

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

        if d < n.mu:
            # Should at least check left child, optionally right if distance
            # radius overlaps right region.
            if n.children[0]:  # d < n.mu + tau
                to_search_push(n.children[0])
            if n.children[1] and d >= n.mu - tau:
                to_search_push(n.children[1])
        else:
            # Should at least check right child, optionally left if distance
            # radius overlaps left region.
            if n.children[1]:  # d >= n.mu - tau
                to_search_push(n.children[1])
            if n.children[0] and d < n.mu + tau:
                to_search_push(n.children[0])

    # Need to negate the negated distances for return
    for n in neighbors:
        n[0] = -n[0]
    dists, neighbors = zip(*sorted(neighbors))
    return neighbors, dists
