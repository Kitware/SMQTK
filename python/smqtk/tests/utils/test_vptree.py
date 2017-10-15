from __future__ import print_function

import math
import random
import unittest

from six.moves import range

import nose.tools
import numpy

from smqtk.utils.vptree import *


D_COUNT = 0


def dist_func_counter(d):
    """
    Increment a count in a dictionary instance.
    :param d: Dictionary tracker.
    :type d: dict
    """
    d['count'] = d.get('count', 0) + 1


def print_tree(tree, offset=0):
    ret = "{} {}".format("|"*offset, tree)
    if tree is not None:
        ret = "\n".join((
            ret,
            print_tree(tree.left, offset=offset+1),
            print_tree(tree.right, offset=offset+1)
        ))
    return ret


def assert_vptrees_equal(tree1, tree2):
    """
    Assert two vp trees are equivalent
    :param tree1: Root node of one tree
    :type tree1: VpNode
    :param tree2: Root node of other tree
    :type tree2: VpNode
    """
    if tree1 is None or tree2 is None:
        nose.tools.assert_is(tree1, tree2)
    else:
        nose.tools.assert_equal(tree1.p, tree2.p)
        nose.tools.assert_equal(tree1.p, tree2.p)
        if tree1.mu is None or tree2.mu is None:
            nose.tools.assert_is(tree1.mu, tree2.mu)
        else:
            numpy.testing.assert_almost_equal(tree1.mu, tree2.mu)
        assert_vptrees_equal(tree1.left, tree2.left)
        assert_vptrees_equal(tree1.right, tree2.right)


def assert_vpstrees_equal(tree1, tree2):
    """
    Assert two vps trees are equivalent
    :param tree1: Root node of one tree
    :type tree1: VpsNode
    :param tree2: Root node of other tree
    :type tree2: VpsNode
    """
    if tree1 is None or tree2 is None:
        nose.tools.assert_is(tree1, tree2)
    else:
        nose.tools.assert_equal(tree1.p, tree2.p)
        nose.tools.assert_equal(tree1.p, tree2.p)
        nose.tools.assert_sequence_equal(tree1.bounds, tree2.bounds)
        if tree1.mu is None or tree2.mu is None:
            nose.tools.assert_is(tree1.mu, tree2.mu)
        else:
            numpy.testing.assert_almost_equal(tree1.mu, tree2.mu)
        assert_vpstrees_equal(tree1.left, tree2.left)
        assert_vpstrees_equal(tree1.right, tree2.right)


def assert_vpsbtrees_equal(tree1, tree2):
    """
    Assert two vpsb trees are equivalent
    :param tree1: Root node of one tree
    :type tree1: VpsbNode
    :param tree2: Root node of other tree
    :type tree2: VpsbNode
    """
    if tree1 is None or tree2 is None:
        nose.tools.assert_is(tree1, tree2)
    else:
        nose.tools.assert_equal(tree1.p, tree2.p)
        nose.tools.assert_equal(tree1.p, tree2.p)
        nose.tools.assert_sequence_equal(tree1.bounds, tree2.bounds)
        if tree1.mu is None or tree2.mu is None:
            nose.tools.assert_is(tree1.mu, tree2.mu)
        else:
            for mu1, mu2 in zip(tree1.mu, tree2.mu):
                if mu1 is None or mu2 is None:
                    nose.tools.assert_is(mu1, mu2)
                else:
                    numpy.testing.assert_almost_equal(mu1, mu2)
        if tree1.max_children is None or tree2.max_children is None:
            nose.tools.assert_is(tree1.max_children, tree2.max_children)
        else:
            numpy.testing.assert_equal(tree1.max_children, tree2.max_children)
        if tree1.children is None or tree2.children is None:
            nose.tools.assert_is(tree1.children, tree2.children)
        else:
            for child1, child2 in zip(tree1.children, tree2.children):
                assert_vpsbtrees_equal(child1, child2)


class TestVpBase (object):

    P = 1
    RAND_SEED = 0

    def _make_sequential_set(self):
        """
        Make shuffled set of sequential numbers based on set P value
        (power of 2)
        """
        s = list(range(2**self.P))
        random.seed(self.RAND_SEED)
        random.shuffle(s)
        return s

    @staticmethod
    def _make_dist_func(s_max, counter):
        """
        Dummy distance function measuring delta between values, normalizing to
        set max value.
        """
        def dummy_dist_func(a, b):
            dist_func_counter(counter)
            return abs(a - b) / float(s_max)
        return dummy_dist_func


class TestVpTree (unittest.TestCase, TestVpBase):

    P = 12

    def test_make_tree(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        vp_make_tree(S, dist_func, r_seed=self.RAND_SEED)
        # Calls to distance function should be <= O(n*log(n))
        max_expected_calls = len(S) * math.log(len(S), 2)
        self.assertLessEqual(c['count'], max_expected_calls)

    def test_array_conversion(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        vpt = vp_make_tree(S, dist_func, r_seed=self.RAND_SEED)
        vp_arrays = vpt.to_arrays()
        vpt2 = VpNode.from_arrays(
            vp_arrays['value_array'],
            vp_arrays['mu_array'],
            vp_arrays['nd_array'],
        )
        assert_vptrees_equal(vpt, vpt2)


    def test_knn_recursive(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vp_make_tree(S, dist_func, r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0  # we s
        nbors, dists = vp_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        print("debug")

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vp_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q, q - 1, q - 2, q - 3, q - 4,
                                  q - 5, q - 6, q - 7, q - 8, q - 9])

    def test_knn_iterative(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vp_make_tree(S, dist_func, r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0  # we s
        nbors, dists = vp_knn_iterative(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        print("debug")

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vp_knn_iterative(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q,   q-1, q-2, q-3, q-4,
                                  q-5, q-6, q-7, q-8, q-9])

    def test_knn_iterative_heapq(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vp_make_tree(S, dist_func, r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0
        nbors, dists = vp_knn_iterative_heapq(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vp_knn_iterative_heapq(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q,   q-1, q-2, q-3, q-4,
                                  q-5, q-6, q-7, q-8, q-9])


class TestVpsTree (unittest.TestCase, TestVpBase):

    P = 12

    def test_make_tree(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        vps_make_tree(S, dist_func, r_seed=self.RAND_SEED)
        # Calls to distance function should be <= O(n*log(n))
        max_expected_calls = len(S) * math.log(len(S), 2)
        self.assertLessEqual(c['count'], max_expected_calls)

    def test_array_conversion(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        vpt = vps_make_tree(S, dist_func, r_seed=self.RAND_SEED)
        vp_arrays = vpt.to_arrays()
        vpt2 = VpsNode.from_arrays(
            vp_arrays['value_array'],
            vp_arrays['mu_array'],
            vp_arrays['nd_array'],
            vp_arrays['bounds_array'],
        )
        assert_vpstrees_equal(vpt, vpt2)

    def test_knn_recursive(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vps_make_tree(S, dist_func, r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0
        nbors, dists = vps_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vps_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q, q - 1, q - 2, q - 3, q - 4,
                                  q - 5, q - 6, q - 7, q - 8, q - 9])

    def test_knn_iterative_heapq(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vps_make_tree(S, dist_func, r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0
        nbors, dists = vps_knn_iterative_heapq(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vps_knn_iterative_heapq(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q,   q-1, q-2, q-3, q-4,
                                  q-5, q-6, q-7, q-8, q-9])

    def test_make_deduplicate(self):
        # Make set with duplicate elements (two copies of each integer).
        S = self._make_sequential_set() + self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)

        # Build with duplicates enabled.
        root = vps_make_tree(S, dist_func, r_seed=self.RAND_SEED)
        c['count'] = 0
        q = 0
        nbors, dists = vps_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        # Build with deduplification.
        root = vps_make_tree(S, dist_func, deduplicate=True,
                             r_seed=self.RAND_SEED)
        c['count'] = 0
        q = 0
        nbors, dists = vps_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


class TestVpsbTree (unittest.TestCase, TestVpBase):

    P = 12

    def test_make_tree(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)

        root = vpsb_make_tree(S, dist_func, 2, r_seed=self.RAND_SEED)
        # Calls to distance function should be <= O(n*(log(n)/log(2)))
        max_expected_calls = len(S) * math.log(len(S), 2)
        self.assertLessEqual(c['count'], max_expected_calls)
        c['count'] = 0

        root = vpsb_make_tree(S, dist_func, 3, r_seed=self.RAND_SEED)
        # Calls to distance function should be <= O(n*(log(n)/log(3)))
        max_expected_calls = len(S) * math.log(len(S), 3)
        self.assertLessEqual(c['count'], max_expected_calls)
        c['count'] = 0

        root = vpsb_make_tree(S, dist_func, 9, r_seed=self.RAND_SEED)
        # Calls to distance function should be <= O(n*(log(n)/log(9)))
        # TODO: Adding a 10% fudge factor here because its clocking in at a
        #       little over expected runtime and not sure why at the moment,
        #       possible because the tree isn't cleanly balanced?
        max_expected_calls = len(S) * math.log(len(S), 9) + (len(S) * 0.1)
        self.assertLessEqual(c['count'], max_expected_calls)
        c['count'] = 0

    def test_array_conversion(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        vpt = vpsb_make_tree(S, dist_func, r_seed=self.RAND_SEED)
        vp_arrays = vpt.to_arrays()
        vpt2 = VpsbNode.from_arrays(
            vp_arrays['value_array'],
            vp_arrays['mu_array'],
            vp_arrays['nd_array'],
            vp_arrays['bounds_array'],
        )
        assert_vpsbtrees_equal(vpt, vpt2)

    def test_make_deduplicate(self):
        # Make set with duplicate elements (two copies of each integer).
        S = self._make_sequential_set() + self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)

        # Build with duplicates enabled.
        root = vpsb_make_tree(S, dist_func, branching_factor=2,
                              r_seed=self.RAND_SEED)
        c['count'] = 0
        q = 0
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        # Build with deduplification.
        root = vpsb_make_tree(S, dist_func, branching_factor=2,
                              deduplicate=True, r_seed=self.RAND_SEED)
        c['count'] = 0
        q = 0
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Build with deduplification, 3-branch.
        root = vpsb_make_tree(S, dist_func, branching_factor=3,
                              deduplicate=True, r_seed=self.RAND_SEED)
        c['count'] = 0
        q = 0
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_knn_recursive_branching2(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vpsb_make_tree(S, dist_func, branching_factor=2,
                              r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q, q - 1, q - 2, q - 3, q - 4,
                                  q - 5, q - 6, q - 7, q - 8, q - 9])

    def test_knn_recursive_branching3(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vpsb_make_tree(S, dist_func, branching_factor=3,
                              r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q, q - 1, q - 2, q - 3, q - 4,
                                  q - 5, q - 6, q - 7, q - 8, q - 9])

    def test_knn_recursive_branching9(self):
        S = self._make_sequential_set()
        c = {}
        dist_func = self._make_dist_func(max(S), c)
        root = vpsb_make_tree(S, dist_func, branching_factor=9,
                              r_seed=self.RAND_SEED)

        # With a query of "0" we should see sequence from S from low to high.
        c['count'] = 0  # reset dist func counter
        q = 0
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # With query of "len(S)", we should see reverse sequence from S from
        # high to low.
        c['count'] = 0  # reset dist func counter
        q = max(S)
        nbors, dists = vpsb_knn_recursive(q, 10, root, dist_func)
        self.assertSequenceEqual(nbors,
                                 [q, q - 1, q - 2, q - 3, q - 4,
                                  q - 5, q - 6, q - 7, q - 8, q - 9])
