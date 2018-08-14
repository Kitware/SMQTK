from __future__ import absolute_import, division, print_function

import random
import os.path as osp
import unittest

import numpy as np
from six.moves import range, zip

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms import get_nn_index_impls
from smqtk.algorithms.nn_index.mrpt import MRPTNearestNeighborsIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex


class TestMRPTIndex (unittest.TestCase):

    RAND_SEED = 42

    def _make_inst(self, **kwargs):
        """
        Make an instance of MRPTNearestNeighborsIndex
        """
        if 'random_seed' not in kwargs:
            kwargs.update(random_seed=self.RAND_SEED)
        return MRPTNearestNeighborsIndex(
            MemoryDescriptorIndex(), **kwargs)

    def test_impl_findable(self):
        self.assertIn(MRPTNearestNeighborsIndex.__name__,
                      get_nn_index_impls())

    def test_configuration(self):
        index_filepath = osp.abspath(osp.expanduser('index_filepath'))
        para_filepath = osp.abspath(osp.expanduser('param_fp'))

        # Make configuration based on default
        c = MRPTNearestNeighborsIndex.get_default_config()
        c['index_filepath'] = index_filepath
        c['parameters_filepath'] = para_filepath
        c['descriptor_set']['type'] = 'MemoryDescriptorIndex'

        # Build based on configuration
        index = MRPTNearestNeighborsIndex.from_config(c)
        self.assertEqual(index._index_filepath, index_filepath)
        self.assertEqual(index._index_param_filepath, para_filepath)

        # Test that constructing a new instance from ``index``'s config yields
        # an index with the same configuration (idempotent).
        index2 = MRPTNearestNeighborsIndex.from_config(index.get_config())
        self.assertEqual(index.get_config(), index2.get_config())

    def test_read_only(self):
        v = np.zeros(5, float)
        v[0] = 1.
        d = DescriptorMemoryElement('unit', 0)
        d.set_vector(v)
        test_descriptors = [d]

        index = self._make_inst(read_only=True)
        self.assertRaises(
            ReadOnlyError,
            index.build_index, test_descriptors
        )

    def test_update_index_no_input(self):
        index = self._make_inst()
        self.assertRaises(
            ValueError,
            index.update_index, []
        )

    def test_update_index_new_index(self):
        n = 100
        dim = 8
        d_index = [DescriptorMemoryElement('test', i) for i in range(n)]
        [d.set_vector(np.random.rand(dim)) for d in d_index]

        index = self._make_inst()
        index.update_index(d_index)
        self.assertEqual(index.count(), 100)
        for d in d_index:
            self.assertIn(d, index._descriptor_set)

        # Check that NN can return stuff from the set used.
        # - nearest element to the query element when the query is in the index
        #   should be the query element.
        random.seed(self.RAND_SEED)
        for _ in range(10):
            i = random.randint(0, n-1)
            q = d_index[i]
            n_elems, n_dists = index.nn(q)
            self.assertEqual(n_elems[0], q)

    def test_update_index_additive(self):
        n1 = 100
        n2 = 10
        dim = 8
        set1 = {DescriptorMemoryElement('test', i) for i in range(n1)}
        set2 = {DescriptorMemoryElement('test', i) for i in range(n1, n1+n2)}
        [d.set_vector(np.random.rand(dim)) for d in set1.union(set1 | set2)]

        # Create and build initial index.
        index = self._make_inst()
        index.build_index(set1)
        self.assertEqual(index.count(), len(set1))
        for d in set1:
            self.assertIn(d, index._descriptor_set)

        # Update and check that all intended descriptors are present in index.
        index.update_index(set2)
        set_all = set1 | set2
        self.assertEqual(index.count(), len(set_all))
        for d in set_all:
            self.assertIn(d, index._descriptor_set)

        # Check that NN can return something from the updated set.
        # - nearest element to the query element when the query is in the index
        #   should be the query element.
        for q in set2:
            n_elems, n_dists = index.nn(q)
            self.assertEqual(n_elems[0], q)

    def test_remove_from_index_readonly(self):
        """
        Test that remove causes an error in a readonly instance.
        """
        index = self._make_inst(read_only=True)
        self.assertRaises(
            ReadOnlyError,
            index.remove_from_index, [0]
        )

    def test_remove_from_index_invalid_uid(self):
        """
        Test that error occurs when attempting to remove descriptor UID that
        isn't indexed.
        """
        index = self._make_inst()
        self.assertRaises(
            KeyError,
            index.remove_from_index, [0]
        )

    def test_remove_from_index(self):
        """
        Test expected removal from the index.
        """
        n = 100
        dim = 32
        dset = [DescriptorMemoryElement('test', i) for i in range(n)]
        np.random.seed(self.RAND_SEED)
        [d.set_vector(np.random.rand(dim)) for d in dset]

        index = self._make_inst()
        index.build_index(dset)
        # Test expected initial condition.
        # noinspection PyCompatibility
        self.assertSetEqual(set(index._descriptor_set.iterkeys()),
                            set(range(100)))

        # Try removing some elements.
        d_to_remove = [dset[10], dset[47], dset[82]]
        index.remove_from_index([d.uuid() for d in d_to_remove])

        # Internal descriptor-set should no longer contain the removed
        # descriptor elements.
        for d in d_to_remove:
            self.assertNotIn(d, index._descriptor_set)
        self.assertEqual(len(index._descriptor_set), 97)
        # Make sure that when we query for elements removed, they are not the
        # returned set of things.
        self.assertNotIn(dset[10].uuid(),
                         set(d.uuid() for d in index.nn(dset[10], n)[0]))
        self.assertNotIn(dset[47].uuid(),
                         set(d.uuid() for d in index.nn(dset[10], n)[0]))
        self.assertNotIn(dset[82].uuid(),
                         set(d.uuid() for d in index.nn(dset[10], n)[0]))

    def test_nn_many_descriptors(self):
        np.random.seed(0)

        n = 10 ** 4
        dim = 256
        depth = 5
        num_trees = 10

        d_index = [DescriptorMemoryElement('test', i) for i in range(n)]
        [d.set_vector(np.random.rand(dim)) for d in d_index]
        q = DescriptorMemoryElement('q', -1)
        q.set_vector(np.zeros((dim,)))

        di = MemoryDescriptorIndex()
        mrpt = MRPTNearestNeighborsIndex(
            di, num_trees=num_trees, depth=depth, random_seed=0)
        mrpt.build_index(d_index)

        nbrs, dists = mrpt.nn(q, 10)
        self.assertEqual(len(nbrs), len(dists))
        self.assertEqual(len(nbrs), 10)

    def test_nn_small_leaves(self):
        np.random.seed(0)

        n = 10 ** 4
        dim = 256
        depth = 10
        # L ~ n/2**depth = 10^4 / 2^10 ~ 10
        k = 200
        # 3k/L = 60
        num_trees = 60

        d_index = [DescriptorMemoryElement('test', i) for i in range(n)]
        [d.set_vector(np.random.rand(dim)) for d in d_index]
        q = DescriptorMemoryElement('q', -1)
        q.set_vector(np.zeros((dim,)))

        di = MemoryDescriptorIndex()
        mrpt = MRPTNearestNeighborsIndex(
            di, num_trees=num_trees, depth=depth, random_seed=0)
        mrpt.build_index(d_index)

        nbrs, dists = mrpt.nn(q, k)
        self.assertEqual(len(nbrs), len(dists))
        self.assertEqual(len(nbrs), k)

    def test_nn_pathological_example(self):
        n = 10 ** 4
        dim = 256
        depth = 10
        # L ~ n/2**depth = 10^4 / 2^10 ~ 10
        k = 200
        # 3k/L = 60
        num_trees = 60

        d_index = [DescriptorMemoryElement('test', i) for i in range(n)]
        # Put all descriptors on a line so that different trees get same
        # divisions.
        # noinspection PyTypeChecker
        [d.set_vector(np.full(dim, d.uuid(), dtype=np.float64))
         for d in d_index]
        q = DescriptorMemoryElement('q', -1)
        q.set_vector(np.zeros((dim,)))

        di = MemoryDescriptorIndex()
        mrpt = MRPTNearestNeighborsIndex(
            di, num_trees=num_trees, depth=depth, random_seed=0)
        mrpt.build_index(d_index)

        nbrs, dists = mrpt.nn(q, k)
        self.assertEqual(len(nbrs), len(dists))
        # We should get about 10 descriptors back instead of the requested
        # 200
        self.assertLess(len(nbrs), 20)

    def test_nn_known_descriptors_euclidean_unit(self):
        dim = 5

        ###
        # Unit vectors -- Equal distance
        #
        index = self._make_inst()
        test_descriptors = []
        for i in range(dim):
            v = np.zeros(dim, float)
            v[i] = 1.
            d = DescriptorMemoryElement('unit', i)
            d.set_vector(v)
            test_descriptors.append(d)
        index.build_index(test_descriptors)
        # query descriptor -- zero vector
        # -> all modeled descriptors should be equally distant (unit
        # corners)
        q = DescriptorMemoryElement('query', 0)
        q.set_vector(np.zeros(dim, float))
        r, dists = index.nn(q, n=dim)
        self.assertEqual(len(dists), dim)
        # All dists should be 1.0, r order doesn't matter
        for d in dists:
            self.assertEqual(d, 1.)

    def test_nn_known_descriptors_nearest(self):
        dim = 5

        ###
        # Unit vectors -- Equal distance
        #
        index = self._make_inst()
        test_descriptors = []
        vectors = np.eye(dim, dtype=np.float32)
        for i in range(dim):
            d = DescriptorMemoryElement('unit', i)
            d.set_vector(vectors[i])
            test_descriptors.append(d)
        index.build_index(test_descriptors)
        for i in range(dim):
            # query descriptor -- first point
            q = DescriptorMemoryElement('query', i)
            q.set_vector(vectors[i])
            r, dists = index.nn(q)
            self.assertEqual(len(dists), 1)
            # Distance should be zero (exact match)
            self.assertEqual(dists[0], 0.)
            np.testing.assert_allclose(r[0].vector(), vectors[i])

    def test_nn_known_descriptors_euclidean_ordered(self):
        index = self._make_inst()

        # make vectors to return in a known euclidean distance order
        i = 100
        test_descriptors = []
        for j in range(i):
            d = DescriptorMemoryElement('ordered', j)
            d.set_vector(np.array([j, j*2], float))
            test_descriptors.append(d)
        random.shuffle(test_descriptors)
        index.build_index(test_descriptors)

        # Since descriptors were build in increasing distance from (0,0),
        # returned descriptors for a query of [0,0] should be in index
        # order.
        q = DescriptorMemoryElement('query', 99)
        q.set_vector(np.array([0, 0], float))
        r, dists = index.nn(q, n=i)
        # Because the data is one-dimensional, all of the cells will have
        # the same points (any division will just correspond to a point on
        # the line), and a cell can't have more than half of the points
        self.assertEqual(len(dists), i//2)
        for j, d, dist in zip(range(i), r, dists):
            self.assertEqual(d.uuid(), j)
            np.testing.assert_equal(d.vector(), [j, j*2])
