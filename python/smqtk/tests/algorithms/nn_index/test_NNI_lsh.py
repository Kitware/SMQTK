from __future__ import division, print_function

import collections
import json
import mock
import random
import types
import unittest

import numpy as np

from smqtk.algorithms.nn_index.lsh import LSHNearestNeighborIndex
from smqtk.algorithms.nn_index.lsh.functors import LshFunctor
from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.algorithms.nn_index.hash_index import HashIndex
from smqtk.algorithms.nn_index.hash_index.linear import LinearHashIndex
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex
from smqtk.representation.key_value.memory import MemoryKeyValueStore


class DummyHashFunctor (LshFunctor):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        """ stub """

    def get_hash(self, descriptor):
        """
        Dummy function that returns the bits of the integer sum of descriptor
        vector.

        :param descriptor: Descriptor vector we should generate the hash of.
        :type descriptor: np.ndarray[float]

        :return: Generated bit-vector as a numpy array of booleans.
        :rtype: np.ndarray[bool]

        """
        return np.asarray([int(c) for c in bin(int(descriptor.sum()))[2:]],
                             bool)


class TestLshIndex (unittest.TestCase):

    def test_is_usable(self):
        # Should always be usable since this is a shell class.
        self.assertTrue(LSHNearestNeighborIndex.is_usable())

    def test_configuration(self):
        c = LSHNearestNeighborIndex.get_default_config()

        # Check that default is in JSON format and is decoded to the same
        # result.
        self.assertEqual(json.loads(json.dumps(c)), c)

        # Make a simple configuration
        # - ItqFunctor should always be available since it has no dependencies.
        c['lsh_functor']['type'] = 'ItqFunctor'
        c['descriptor_index']['type'] = 'MemoryDescriptorIndex'
        c['hash2uuids_kvstore']['type'] = 'MemoryKeyValueStore'
        c['hash_index']['type'] = 'LinearHashIndex'
        index = LSHNearestNeighborIndex.from_config(c)

        self.assertIsInstance(index.lsh_functor, ItqFunctor)
        self.assertIsInstance(index.descriptor_index, MemoryDescriptorIndex)
        self.assertIsInstance(index.hash_index, LinearHashIndex)
        self.assertIsInstance(index.hash2uuids_kvstore, MemoryKeyValueStore)

        # Can convert instance config to JSON
        self.assertEqual(
            json.loads(json.dumps(index.get_config())),
            index.get_config()
        )

    def test_configuration_none_HI(self):
        c = LSHNearestNeighborIndex.get_default_config()

        # Check that default is in JSON format and is decoded to the same
        # result.
        self.assertEqual(json.loads(json.dumps(c)), c)

        # Make a simple configuration
        c['lsh_functor']['type'] = 'ItqFunctor'
        c['descriptor_index']['type'] = 'MemoryDescriptorIndex'
        c['hash2uuids_kvstore']['type'] = 'MemoryKeyValueStore'
        c['hash_index']['type'] = None
        index = LSHNearestNeighborIndex.from_config(c)

        self.assertIsInstance(index.lsh_functor, ItqFunctor)
        self.assertIsInstance(index.descriptor_index, MemoryDescriptorIndex)
        self.assertIsNone(index.hash_index)
        self.assertIsInstance(index.hash2uuids_kvstore, MemoryKeyValueStore)

        # Can convert instance config to JSON
        self.assertEqual(
            json.loads(json.dumps(index.get_config())),
            index.get_config()
        )

    def test_get_dist_func_euclidean(self):
        f = LSHNearestNeighborIndex._get_dist_func('euclidean')
        self.assertIsInstance(f, types.FunctionType)
        self.assertAlmostEqual(
            f(np.array([0, 0]), np.array([0, 1])),
            1.0
        )

    def test_get_dist_func_cosine(self):
        f = LSHNearestNeighborIndex._get_dist_func('cosine')
        self.assertIsInstance(f, types.FunctionType)
        self.assertAlmostEqual(
            f(np.array([1, 0]), np.array([0, 1])),
            1.0
        )
        self.assertAlmostEqual(
            f(np.array([1, 0]), np.array([1, 1])),
            0.5
        )

    def test_get_dist_func_hik(self):
        f = LSHNearestNeighborIndex._get_dist_func('hik')
        self.assertIsInstance(f, types.FunctionType)
        self.assertAlmostEqual(
            f(np.array([0, 0]), np.array([0, 1])),
            1.0
        )
        self.assertAlmostEqual(
            f(np.array([1, 0]), np.array([0, 1])),
            1.0
        )
        self.assertAlmostEqual(
            f(np.array([1, 1]), np.array([0, 1])),
            0.0
        )

    def test_get_dist_func_invalid_string(self):
        self.assertRaises(
            ValueError,
            LSHNearestNeighborIndex._get_dist_func,
            'not-valid-string'
        )

    def test_count_empty_hash2uid(self):
        """
        Test that an empty hash-to-uid mapping results in a 0 return regardless
        of descriptor-set state.
        """
        descr_set = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        self.assertEqual(descr_set.count(), 0)
        self.assertEqual(hash_kvs.count(), 0)

        lsh = LSHNearestNeighborIndex(DummyHashFunctor(), descr_set, hash_kvs)
        self.assertEqual(lsh.count(), 0)

        # Additions to the descriptor-set should not impact LSH index "size"
        lsh.descriptor_index.add_descriptor(DescriptorMemoryElement('t', 0))
        self.assertEqual(lsh.descriptor_index.count(), 1)
        self.assertEqual(lsh.hash2uuids_kvstore.count(), 0)
        self.assertEqual(lsh.count(), 0)

        lsh.descriptor_index.add_descriptor(DescriptorMemoryElement('t', 1))
        self.assertEqual(lsh.descriptor_index.count(), 2)
        self.assertEqual(lsh.hash2uuids_kvstore.count(), 0)
        self.assertEqual(lsh.count(), 0)

        lsh.hash2uuids_kvstore.add(0, {0})
        self.assertEqual(lsh.descriptor_index.count(), 2)
        self.assertEqual(lsh.count(), 1)

        lsh.hash2uuids_kvstore.add(0, {0, 1})
        self.assertEqual(lsh.descriptor_index.count(), 2)
        self.assertEqual(lsh.count(), 2)

        lsh.hash2uuids_kvstore.add(0, {0, 1, 2})
        self.assertEqual(lsh.descriptor_index.count(), 2)
        self.assertEqual(lsh.count(), 3)

    def test_build_index_read_only(self):
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        MemoryDescriptorIndex(),
                                        MemoryKeyValueStore(), read_only=True)
        self.assertRaises(
            ReadOnlyError,
            index._build_index, []
        )

    def test_build_index_fresh_build(self):
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs)

        descriptors = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for i, d in enumerate(descriptors):
            d.set_vector(np.ones(1, float) * i)
        index.build_index(descriptors)

        # Make sure descriptors are now in attached index and in key-value-store
        self.assertEqual(descr_index.count(), 5)
        for d in descriptors:
            self.assertIn(d, descr_index)
        # Dummy hash function bins sum of descriptor vectors.
        self.assertEqual(hash_kvs.count(), 5)
        for i in range(5):
            self.assertSetEqual(hash_kvs.get(i), {i})

    def test_build_index_fresh_build_with_hash_index(self):
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        linear_hi = LinearHashIndex()  # simplest hash index, heap-sorts.
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs, linear_hi)

        descriptors = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for i, d in enumerate(descriptors):
            d.set_vector(np.ones(1, float) * i)
        index.build_index(descriptors)
        # Hash index should have been built with hash vectors, and linearHI
        # converts those to integers for storage.
        self.assertEqual(linear_hi.index, {0, 1, 2, 3, 4})

    def test_update_index_read_only(self):
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        MemoryDescriptorIndex(),
                                        MemoryKeyValueStore(), read_only=True)
        self.assertRaises(
            ReadOnlyError,
            index._update_index, []
        )

    def test_update_index_no_existing_index(self):
        # Test that calling update_index with no existing index acts like
        # building the index fresh.  This test is basically the same as
        # test_build_index_fresh_build but using update_index instead.
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs)

        descriptors = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for d in descriptors:
            d.set_vector(np.ones(1, float) * d.uuid())
        index.update_index(descriptors)

        # Make sure descriptors are now in attached index and in key-value-store
        self.assertEqual(descr_index.count(), 5)
        for d in descriptors:
            self.assertIn(d, descr_index)
        # Dummy hash function bins sum of descriptor vectors.
        self.assertEqual(hash_kvs.count(), 5)
        for i in range(5):
            self.assertSetEqual(hash_kvs.get(i), {i})

    def test_update_index_add_new_descriptors(self):
        # Test that calling update index after a build index causes index
        # components to be properly updated.
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs)
        descriptors1 = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        descriptors2 = [
            DescriptorMemoryElement('t', 5),
            DescriptorMemoryElement('t', 6),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for d in descriptors1 + descriptors2:
            d.set_vector(np.ones(1, float) * d.uuid())

        # Build initial index.
        index.build_index(descriptors1)
        self.assertEqual(descr_index.count(), 5)
        for d in descriptors1:
            self.assertIn(d, descr_index)
        for d in descriptors2:
            self.assertNotIn(d, descr_index)
        # Dummy hash function bins sum of descriptor vectors.
        self.assertEqual(hash_kvs.count(), 5)
        for i in range(5):
            self.assertSetEqual(hash_kvs.get(i), {i})

        # Update index and check that components have new data.
        index.update_index(descriptors2)
        self.assertEqual(descr_index.count(), 7)
        for d in descriptors1 + descriptors2:
            self.assertIn(d, descr_index)
        # Dummy hash function bins sum of descriptor vectors.
        self.assertEqual(hash_kvs.count(), 7)
        for i in range(7):
            self.assertSetEqual(hash_kvs.get(i), {i})

    def test_update_index_duplicate_descriptors(self):
        """
        Test that updating a built index with the same descriptors results in
        idempotent behavior.
        """
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs)

        # Identical Descriptors to build and update on (different instances)
        descriptors1 = [
            DescriptorMemoryElement('t', 0).set_vector([0]),
            DescriptorMemoryElement('t', 1).set_vector([1]),
            DescriptorMemoryElement('t', 2).set_vector([2]),
            DescriptorMemoryElement('t', 3).set_vector([3]),
            DescriptorMemoryElement('t', 4).set_vector([4]),
        ]
        descriptors2 = [
            DescriptorMemoryElement('t', 0).set_vector([0]),
            DescriptorMemoryElement('t', 1).set_vector([1]),
            DescriptorMemoryElement('t', 2).set_vector([2]),
            DescriptorMemoryElement('t', 3).set_vector([3]),
            DescriptorMemoryElement('t', 4).set_vector([4]),
        ]

        index.build_index(descriptors1)
        index.update_index(descriptors2)

        assert descr_index.count() == 5
        # Above descriptors should be considered "in" the descriptor set now.
        for d in descriptors1:
            assert d in descr_index
        for d in descriptors2:
            assert d in descr_index
        # Known hashes of the above descriptors should be in the KVS
        assert set(hash_kvs.keys()) == {0, 1, 2, 3, 4}
        assert hash_kvs.get(0) == {0}
        assert hash_kvs.get(1) == {1}
        assert hash_kvs.get(2) == {2}
        assert hash_kvs.get(3) == {3}
        assert hash_kvs.get(4) == {4}

    def test_update_index_similar_descriptors(self):
        """
        Test that updating a built index with similar descriptors (same
        vectors, different UUIDs) results in contained structures having an
        expected state.
        """
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs)

        # Similar Descriptors to build and update on (different instances)
        descriptors1 = [
            DescriptorMemoryElement('t', 0).set_vector([0]),
            DescriptorMemoryElement('t', 1).set_vector([1]),
            DescriptorMemoryElement('t', 2).set_vector([2]),
            DescriptorMemoryElement('t', 3).set_vector([3]),
            DescriptorMemoryElement('t', 4).set_vector([4]),
        ]
        descriptors2 = [
            DescriptorMemoryElement('t', 5).set_vector([0]),
            DescriptorMemoryElement('t', 6).set_vector([1]),
            DescriptorMemoryElement('t', 7).set_vector([2]),
            DescriptorMemoryElement('t', 8).set_vector([3]),
            DescriptorMemoryElement('t', 9).set_vector([4]),
        ]

        index.build_index(descriptors1)
        index.update_index(descriptors2)

        assert descr_index.count() == 10
        # Above descriptors should be considered "in" the descriptor set now.
        for d in descriptors1:
            assert d in descr_index
        for d in descriptors2:
            assert d in descr_index
        # Known hashes of the above descriptors should be in the KVS
        assert set(hash_kvs.keys()) == {0, 1, 2, 3, 4}
        assert hash_kvs.get(0) == {0, 5}
        assert hash_kvs.get(1) == {1, 6}
        assert hash_kvs.get(2) == {2, 7}
        assert hash_kvs.get(3) == {3, 8}
        assert hash_kvs.get(4) == {4, 9}

    def test_update_index_existing_descriptors_frozenset(self):
        """
        Same as ``test_update_index_similar_descriptors`` but testing that
        we can update the index when seeded with structures with existing
        values.
        """
        # Similar Descriptors to build and update on (different instances)
        descriptors1 = [
            DescriptorMemoryElement('t', 0).set_vector([0]),
            DescriptorMemoryElement('t', 1).set_vector([1]),
            DescriptorMemoryElement('t', 2).set_vector([2]),
            DescriptorMemoryElement('t', 3).set_vector([3]),
            DescriptorMemoryElement('t', 4).set_vector([4]),
        ]
        descriptors2 = [
            DescriptorMemoryElement('t', 5).set_vector([0]),
            DescriptorMemoryElement('t', 6).set_vector([1]),
            DescriptorMemoryElement('t', 7).set_vector([2]),
            DescriptorMemoryElement('t', 8).set_vector([3]),
            DescriptorMemoryElement('t', 9).set_vector([4]),
        ]

        descr_index = MemoryDescriptorIndex()
        descr_index.add_many_descriptors(descriptors1)

        hash_kvs = MemoryKeyValueStore()
        hash_kvs.add(0, frozenset({0}))
        hash_kvs.add(1, frozenset({1}))
        hash_kvs.add(2, frozenset({2}))
        hash_kvs.add(3, frozenset({3}))
        hash_kvs.add(4, frozenset({4}))

        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs)
        index.update_index(descriptors2)

        assert descr_index.count() == 10
        # Above descriptors should be considered "in" the descriptor set now.
        for d in descriptors1:
            assert d in descr_index
        for d in descriptors2:
            assert d in descr_index
        # Known hashes of the above descriptors should be in the KVS
        assert set(hash_kvs.keys()) == {0, 1, 2, 3, 4}
        assert hash_kvs.get(0) == {0, 5}
        assert hash_kvs.get(1) == {1, 6}
        assert hash_kvs.get(2) == {2, 7}
        assert hash_kvs.get(3) == {3, 8}
        assert hash_kvs.get(4) == {4, 9}

    def test_update_index_with_hash_index(self):
        # Similar test to `test_update_index_add_new_descriptors` but with a
        # linear hash index.
        descr_index = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        linear_hi = LinearHashIndex()  # simplest hash index, heap-sorts.
        index = LSHNearestNeighborIndex(DummyHashFunctor(),
                                        descr_index, hash_kvs, linear_hi)

        descriptors1 = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        descriptors2 = [
            DescriptorMemoryElement('t', 5),
            DescriptorMemoryElement('t', 6),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for d in descriptors1 + descriptors2:
            d.set_vector(np.ones(1, float) * d.uuid())

        # Build initial index.
        index.build_index(descriptors1)
        # Initial hash index should only encode hashes for first batch of
        # descriptors.
        self.assertSetEqual(linear_hi.index, {0, 1, 2, 3, 4})

        # Update index and check that components have new data.
        index.update_index(descriptors2)
        # Now the hash index should include all descriptor hashes.
        self.assertSetEqual(linear_hi.index, {0, 1, 2, 3, 4, 5, 6})

    def test_remove_from_index_read_only(self):
        d_set = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        idx = LSHNearestNeighborIndex(DummyHashFunctor(), d_set, hash_kvs,
                                      read_only=True)
        self.assertRaises(
            ReadOnlyError,
            idx.remove_from_index,
            ['uid1', 'uid2']
        )

    def test_remove_from_index_no_existing_index(self):
        # Test that attempting to remove from an instance with no existing
        # index (meaning empty descriptor-set and key-value-store) results in
        # a key error.
        d_set = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        idx = LSHNearestNeighborIndex(DummyHashFunctor(), d_set, hash_kvs)
        self.assertRaisesRegexp(
            KeyError,
            'uid1',
            idx.remove_from_index,
            ['uid1']
        )

    def test_remove_from_index_invalid_uid(self):
        # Test that attempting to remove a single invalid UID causes a key
        # error and does not affect index.

        # Descriptors are 1 dim, value == index.
        descriptors = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for d in descriptors:
            d.set_vector(np.ones(1, float) * d.uuid())
        # uid -> descriptor
        expected_dset_table = {
            0: descriptors[0],
            1: descriptors[1],
            2: descriptors[2],
            3: descriptors[3],
            4: descriptors[4],
        }
        # hash int -> set[uid]
        expected_kvs_table = {
            0: {0},
            1: {1},
            2: {2},
            3: {3},
            4: {4},
        }

        d_set = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        idx = LSHNearestNeighborIndex(DummyHashFunctor(), d_set, hash_kvs)
        idx.build_index(descriptors)
        # Assert we have the correct expected values
        self.assertEqual(idx.descriptor_index._table, expected_dset_table)
        self.assertEqual(idx.hash2uuids_kvstore._table, expected_kvs_table)

        # Attempt to remove descriptor with a UID we did not build with.
        self.assertRaisesRegexp(
            KeyError, '5',
            idx.remove_from_index, [5]
        )
        # Index should not have been modified.
        self.assertEqual(idx.descriptor_index._table, expected_dset_table)
        self.assertEqual(idx.hash2uuids_kvstore._table, expected_kvs_table)

        # Attempt to remove multiple UIDs, one valid and one invalid
        self.assertRaisesRegexp(
            KeyError, '5',
            idx.remove_from_index, [2, 5]
        )
        # Index should not have been modified.
        self.assertEqual(idx.descriptor_index._table, expected_dset_table)
        self.assertEqual(idx.hash2uuids_kvstore._table, expected_kvs_table)

    def test_remove_from_index(self):
        # Test that removing by UIDs does the correct thing.

        # Descriptors are 1 dim, value == index.
        descriptors = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for d in descriptors:
            d.set_vector(np.ones(1, float) * d.uuid())
        d_set = MemoryDescriptorIndex()
        hash_kvs = MemoryKeyValueStore()
        idx = LSHNearestNeighborIndex(DummyHashFunctor(), d_set, hash_kvs)
        idx.build_index(descriptors)

        # Attempt removing 1 uid.
        idx.remove_from_index([3])
        self.assertEqual(idx.descriptor_index._table, {
            0: descriptors[0],
            1: descriptors[1],
            2: descriptors[2],
            4: descriptors[4],
        })
        self.assertEqual(idx.hash2uuids_kvstore._table, {
            0: {0},
            1: {1},
            2: {2},
            4: {4},
        })

    def test_remove_from_index_shared_hashes(self):
        """
        Test that removing a descriptor (by UID) that shares a hash with other
        descriptors does not trigger removal of its hash.
        """
        # Simulate descriptors all hashing to the same hash value: 0
        hash_func = DummyHashFunctor()
        hash_func.get_hash = mock.Mock(return_value=np.asarray([0], bool))

        d_set = MemoryDescriptorIndex()
        hash2uids_kvs = MemoryKeyValueStore()
        idx = LSHNearestNeighborIndex(hash_func, d_set, hash2uids_kvs)

        # Descriptors are 1 dim, value == index.
        descriptors = [
            DescriptorMemoryElement('t', 0),
            DescriptorMemoryElement('t', 1),
            DescriptorMemoryElement('t', 2),
            DescriptorMemoryElement('t', 3),
            DescriptorMemoryElement('t', 4),
        ]
        # Vectors of length 1 for easy dummy hashing prediction.
        for d in descriptors:
            d.set_vector(np.ones(1, float) * d.uuid())
        idx.build_index(descriptors)
        # We expect the descriptor-set and kvs to look like the following now:
        self.assertDictEqual(d_set._table, {
            0: descriptors[0],
            1: descriptors[1],
            2: descriptors[2],
            3: descriptors[3],
            4: descriptors[4],
        })
        self.assertDictEqual(hash2uids_kvs._table, {0: {0, 1, 2, 3, 4}})

        # Mock out hash index as if we had an implementation so we can check
        # call to its remove_from_index method.
        idx.hash_index = mock.Mock(spec=HashIndex)

        idx.remove_from_index([2, 4])

        # Only uid 2 and 4 descriptors should be gone from d-set, kvs should
        # still have the 0 key and its set value should only contain uids 0, 1
        # and 3.  `hash_index.remove_from_index` should not be called because
        # no hashes should be marked for removal.
        self.assertDictEqual(d_set._table, {
            0: descriptors[0],
            1: descriptors[1],
            3: descriptors[3],
        })
        self.assertDictEqual(hash2uids_kvs._table, {0: {0, 1, 3}})
        idx.hash_index.remove_from_index.assert_not_called()

    def test_remove_from_index_shared_hashes_partial(self):
        """
        Test that only some hashes are removed from the hash index, but not
        others when those hashes still refer to other descriptors.
        """
        # Simulate initial state with some descriptor hashed to one value and
        # other descriptors hashed to another.

        # Vectors of length 1 for easy dummy hashing prediction.
        descriptors = [
            DescriptorMemoryElement('t', 0).set_vector([0]),
            DescriptorMemoryElement('t', 1).set_vector([1]),
            DescriptorMemoryElement('t', 2).set_vector([2]),
            DescriptorMemoryElement('t', 3).set_vector([3]),
            DescriptorMemoryElement('t', 4).set_vector([4]),
        ]

        # Dummy hash function to do the simulated thing
        hash_func = DummyHashFunctor()
        hash_func.get_hash = mock.Mock(
            # Vectors of even sum hash to 0, odd to 1.
            side_effect=lambda vec: [vec.sum() % 2]
        )

        d_set = MemoryDescriptorIndex()
        d_set._table = {
            0: descriptors[0],
            1: descriptors[1],
            2: descriptors[2],
            3: descriptors[3],
            4: descriptors[4],
        }

        hash2uid_kvs = MemoryKeyValueStore()
        hash2uid_kvs._table = {
            0: {0, 2, 4},
            1: {1, 3},
        }

        idx = LSHNearestNeighborIndex(hash_func, d_set, hash2uid_kvs)
        idx.hash_index = mock.Mock(spec=HashIndex)

        idx.remove_from_index([1, 2, 3])
        # Check that only one hash vector was passed to hash_index's removal
        # method (deque of hash-code vectors).
        idx.hash_index.remove_from_index.assert_called_once_with(
            collections.deque([
                [1],
            ])
        )
        self.assertDictEqual(d_set._table, {
            0: descriptors[0],
            4: descriptors[4],
        })
        self.assertDictEqual(hash2uid_kvs._table, {0: {0, 4}})


class TestLshIndexAlgorithms (unittest.TestCase):
    """
    Various tests on the ``nn`` method for different inputs and parameters.
    """

    RANDOM_SEED = 0

    def _make_ftor_itq(self, bits=32):
        itq_ftor = ItqFunctor(bit_length=bits, random_seed=self.RANDOM_SEED)

        def itq_fit(D):
            itq_ftor.fit(D)

        return itq_ftor, itq_fit

    # noinspection PyMethodMayBeStatic
    def _make_hi_linear(self):
        return LinearHashIndex()

    def _make_hi_balltree(self):
        return SkLearnBallTreeHashIndex(random_seed=self.RANDOM_SEED)

    #
    # Test LSH with random vectors
    #
    def _random_euclidean(self, hash_ftor, hash_idx,
                          ftor_train_hook=lambda d: None):
        # :param hash_ftor: Hash function class for generating hash codes for
        #   descriptors.
        # :param hash_idx: Hash index instance to use in local LSH algo
        #   instance.
        # :param ftor_train_hook: Function for training functor if necessary.

        # make random descriptors
        i = 1000
        dim = 256
        td = []
        np.random.seed(self.RANDOM_SEED)
        for j in range(i):
            d = DescriptorMemoryElement('random', j)
            d.set_vector(np.random.rand(dim))
            td.append(d)

        ftor_train_hook(td)

        di = MemoryDescriptorIndex()
        kvstore = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(hash_ftor, di, kvstore,
                                        hash_index=hash_idx,
                                        distance_method='euclidean')
        index.build_index(td)

        # test query from build set -- should return same descriptor when k=1
        q = td[255]
        r, dists = index.nn(q, 1)
        self.assertEqual(r[0], q)

        # test query very near a build vector
        td_q = td[0]
        q = DescriptorMemoryElement('query', i)
        v = td_q.vector().copy()
        v_min = max(v.min(), 0.1)
        v[0] += v_min
        v[dim-1] -= v_min
        q.set_vector(v)
        r, dists = index.nn(q, 1)
        self.assertFalse(np.array_equal(q.vector(), td_q.vector()))
        self.assertEqual(r[0], td_q)

        # random query
        q = DescriptorMemoryElement('query', i+1)
        q.set_vector(np.random.rand(dim))

        # for any query of size k, results should at least be in distance order
        r, dists = index.nn(q, 10)
        for j in range(1, len(dists)):
            self.assertGreater(dists[j], dists[j-1])
        r, dists = index.nn(q, i)
        for j in range(1, len(dists)):
            self.assertGreater(dists[j], dists[j-1])

    def test_random_euclidean__itq__None(self):
        ftor, fit = self._make_ftor_itq()
        self._random_euclidean(ftor, None, fit)

    def test_random_euclidean__itq__linear(self):
        ftor, fit = self._make_ftor_itq()
        hi = self._make_hi_linear()
        self._random_euclidean(ftor, hi, fit)

    def test_random_euclidean__itq__balltree(self):
        ftor, fit = self._make_ftor_itq()
        hi = self._make_hi_balltree()
        self._random_euclidean(ftor, hi, fit)

    #
    # Test unit vectors
    #
    def _known_unit(self, hash_ftor, hash_idx, dist_method,
                    ftor_train_hook=lambda d: None):
        ###
        # Unit vectors - Equal distance
        #
        dim = 5
        test_descriptors = []
        for i in range(dim):
            v = np.zeros(dim, float)
            v[i] = 1.
            d = DescriptorMemoryElement('unit', i)
            d.set_vector(v)
            test_descriptors.append(d)

        ftor_train_hook(test_descriptors)

        di = MemoryDescriptorIndex()
        kvstore = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(hash_ftor, di, kvstore,
                                        hash_index=hash_idx,
                                        distance_method=dist_method)
        index.build_index(test_descriptors)

        # query with zero vector
        # -> all modeled descriptors have no intersection, dists should be 1.0,
        #    or maximum distance by histogram intersection
        q = DescriptorMemoryElement('query', 0)
        q.set_vector(np.zeros(dim, float))
        r, dists = index.nn(q, dim)
        # All dists should be 1.0, r order doesn't matter
        for d in dists:
            self.assertEqual(d, 1.)

        # query with index element
        q = test_descriptors[3]
        r, dists = index.nn(q, 1)
        self.assertEqual(r[0], q)
        self.assertEqual(dists[0], 0.)

        r, dists = index.nn(q, dim)
        self.assertEqual(r[0], q)
        self.assertEqual(dists[0], 0.)

    def test_known_unit__euclidean__itq__None(self):
        ftor, fit = self._make_ftor_itq(5)
        self._known_unit(ftor, None, 'euclidean', fit)

    def test_known_unit__hik__itq__None(self):
        ftor, fit = self._make_ftor_itq(5)
        self._known_unit(ftor, None, 'hik', fit)

    def test_known_unit__euclidean__itq__linear(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_linear()
        self._known_unit(ftor, hi, 'euclidean', fit)

    def test_known_unit__hik__itq__linear(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_linear()
        self._known_unit(ftor, hi, 'hik', fit)

    def test_known_unit__euclidean__itq__balltree(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_balltree()
        self._known_unit(ftor, hi, 'euclidean', fit)

    def test_known_unit__hik__itq__balltree(self):
        ftor, fit = self._make_ftor_itq(5)
        hi = self._make_hi_balltree()
        self._known_unit(ftor, hi, 'hik', fit)

    #
    # Test with known vectors and euclidean dist
    #
    def _known_ordered_euclidean(self, hash_ftor, hash_idx,
                                 ftor_train_hook=lambda d: None):
        # make vectors to return in a known euclidean distance order
        i = 1000
        test_descriptors = []
        for j in range(i):
            d = DescriptorMemoryElement('ordered', j)
            d.set_vector(np.array([j, j*2], float))
            test_descriptors.append(d)
        random.shuffle(test_descriptors)

        ftor_train_hook(test_descriptors)

        di = MemoryDescriptorIndex()
        kvstore = MemoryKeyValueStore()
        index = LSHNearestNeighborIndex(hash_ftor, di, kvstore,
                                        hash_index=hash_idx,
                                        distance_method='euclidean')
        index.build_index(test_descriptors)

        # Since descriptors were built in increasing distance from (0,0),
        # returned descriptors for a query of [0,0] should be in index order.
        q = DescriptorMemoryElement('query', i)
        q.set_vector(np.array([0, 0], float))
        # top result should have UUID == 0 (nearest to query)
        r, dists = index.nn(q, 5)
        self.assertEqual(r[0].uuid(), 0)
        self.assertEqual(r[1].uuid(), 1)
        self.assertEqual(r[2].uuid(), 2)
        self.assertEqual(r[3].uuid(), 3)
        self.assertEqual(r[4].uuid(), 4)
        # global search should be in complete order
        r, dists = index.nn(q, i)
        for j, d, dist in zip(range(i), r, dists):
            self.assertEqual(d.uuid(), j)

    def test_known_ordered_euclidean__itq__None(self):
        ftor, fit = self._make_ftor_itq(1)
        self._known_ordered_euclidean(ftor, None, fit)

    def test_known_ordered_euclidean__itq__linear(self):
        ftor, fit = self._make_ftor_itq(1)
        hi = self._make_hi_linear()
        self._known_ordered_euclidean(ftor, hi, fit)

    def test_known_ordered_euclidean__itq__balltree(self):
        ftor, fit = self._make_ftor_itq(1)
        hi = self._make_hi_balltree()
        self._known_ordered_euclidean(ftor, hi, fit)
