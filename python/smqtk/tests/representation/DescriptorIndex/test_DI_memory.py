import unittest

import numpy
import six
from six.moves import cPickle as pickle

from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex
from smqtk.utils import merge_dict


RAND_UUID = 0


def random_descriptor():
    global RAND_UUID
    d = DescriptorMemoryElement('random', RAND_UUID)
    d.set_vector(numpy.random.rand(64))
    RAND_UUID += 1
    return d


class TestMemoryDescriptorIndex (unittest.TestCase):

    def test_is_usable(self):
        # Always usable because no dependencies.
        self.assertEqual(MemoryDescriptorIndex.is_usable(), True)

    def test_default_config(self):
        # Default should be valid for constructing a new instance.
        c = MemoryDescriptorIndex.get_default_config()
        self.assertEqual(MemoryDescriptorIndex.from_config(c).get_config(), c)

    def test_from_config_null_cache_elem(self):
        inst = MemoryDescriptorIndex.from_config({'cache_element': None})
        self.assertIsNone(inst.cache_element)
        self.assertEqual(inst._table, {})

        inst = MemoryDescriptorIndex.from_config({
            'cache_element': {
                'type': None
            }
        })
        self.assertIsNone(inst.cache_element)
        self.assertEqual(inst._table, {})

    def test_from_config_null_cache_elem_type(self):
        # An empty cache should not trigger loading on construction.
        expected_empty_cache = DataMemoryElement()
        inst = MemoryDescriptorIndex.from_config({
            'cache_element': {
                'type': 'DataMemoryElement',
                'DataMemoryElement': {'bytes': ''}
            }
        })
        self.assertEqual(inst.cache_element, expected_empty_cache)
        self.assertEqual(inst._table, {})

    def test_from_config(self):
        # Configured cache with some picked bytes
        expected_table = dict(a=1, b=2, c=3)
        expected_cache = DataMemoryElement(bytes=pickle.dumps(expected_table))
        inst = MemoryDescriptorIndex.from_config({
            'cache_element': {
                'type': 'DataMemoryElement',
                'DataMemoryElement': {'bytes': expected_cache.get_bytes()}
            }
        })
        self.assertEqual(inst.cache_element, expected_cache)
        self.assertEqual(inst._table, expected_table)

    def test_init_no_cache(self):
        inst = MemoryDescriptorIndex()
        self.assertIsNone(inst.cache_element, None)
        self.assertEqual(inst._table, {})

    def test_init_empty_cache(self):
        cache_elem = DataMemoryElement()
        inst = MemoryDescriptorIndex(cache_element=cache_elem)
        self.assertEqual(inst.cache_element, cache_elem)
        self.assertEqual(inst._table, {})

    def test_init_with_cache(self):
        d_list = (random_descriptor(), random_descriptor(),
                  random_descriptor(), random_descriptor())
        expected_table = dict((r.uuid(), r) for r in d_list)
        expected_cache = DataMemoryElement(bytes=pickle.dumps(expected_table))

        inst = MemoryDescriptorIndex(expected_cache)
        self.assertEqual(len(inst._table), 4)
        self.assertEqual(inst.cache_element, expected_cache)
        self.assertEqual(inst._table, expected_table)
        self.assertEqual(set(inst._table.values()), set(d_list))

    def test_get_config(self):
        self.assertEqual(
            MemoryDescriptorIndex().get_config(),
            MemoryDescriptorIndex.get_default_config()
        )

        self.assertEqual(
            MemoryDescriptorIndex(None).get_config(),
            MemoryDescriptorIndex.get_default_config()
        )

        empty_elem = DataMemoryElement()
        self.assertEqual(
            MemoryDescriptorIndex(empty_elem).get_config(),
            merge_dict(MemoryDescriptorIndex.get_default_config(), {
                'cache_element': {'type': 'DataMemoryElement'}
            })
        )

        dict_pickle_bytes = pickle.dumps({1: 1, 2: 2, 3: 3}, -1)
        cache_elem = DataMemoryElement(bytes=dict_pickle_bytes)
        self.assertEqual(
            MemoryDescriptorIndex(cache_elem).get_config(),
            merge_dict(MemoryDescriptorIndex.get_default_config(), {
                'cache_element': {
                    'DataMemoryElement': {
                        'bytes': dict_pickle_bytes
                    },
                    'type': 'DataMemoryElement'
                }
            })
        )

    def test_cache_table_no_cache(self):
        inst = MemoryDescriptorIndex()
        inst._table = {}
        inst.cache_table()  # should basically do nothing
        self.assertIsNone(inst.cache_element)

    def test_cache_table_empty_table(self):
        inst = MemoryDescriptorIndex(DataMemoryElement(), -1)
        inst._table = {}
        expected_table_pickle_bytes = pickle.dumps(inst._table, -1)

        inst.cache_table()
        self.assertIsNotNone(inst.cache_element)
        self.assertEqual(inst.cache_element.get_bytes(),
                         expected_table_pickle_bytes)

    def test_add_descriptor(self):
        index = MemoryDescriptorIndex()

        d1 = random_descriptor()
        index.add_descriptor(d1)
        self.assertEqual(index._table[d1.uuid()], d1)

        d2 = random_descriptor()
        index.add_descriptor(d2)
        self.assertEqual(index._table[d2.uuid()], d2)

    def test_add_many(self):
        descrs = [
            random_descriptor(),
            random_descriptor(),
            random_descriptor(),
            random_descriptor(),
            random_descriptor(),
        ]
        index = MemoryDescriptorIndex()
        index.add_many_descriptors(descrs)

        # Compare code keys of input to code keys in internal table
        self.assertEqual(set(index._table.keys()),
                         set([e.uuid() for e in descrs]))

        # Get the set of descriptors in the internal table and compare it with
        # the set of generated random descriptors.
        r_set = set()
        [r_set.add(d) for d in index._table.values()]
        self.assertEqual(
            set([e for e in descrs]),
            r_set
        )

    def test_count(self):
        index = MemoryDescriptorIndex()
        self.assertEqual(index.count(), 0)

        d1 = random_descriptor()
        index.add_descriptor(d1)
        self.assertEqual(index.count(), 1)

        d2, d3, d4 = (random_descriptor(),
                      random_descriptor(),
                      random_descriptor())
        index.add_many_descriptors([d2, d3, d4])
        self.assertEqual(index.count(), 4)

        d5 = random_descriptor()
        index.add_descriptor(d5)
        self.assertEqual(index.count(), 5)

    def test_get_descriptors(self):
        descrs = [
            random_descriptor(),   # [0]
            random_descriptor(),   # [1]
            random_descriptor(),   # [2]
            random_descriptor(),   # [3]
            random_descriptor(),   # [4]
        ]
        index = MemoryDescriptorIndex()
        index.add_many_descriptors(descrs)

        # single descriptor reference
        r = index.get_descriptor(descrs[1].uuid())
        self.assertEqual(r, descrs[1])

        # multiple descriptor reference
        r = list(index.get_many_descriptors([descrs[0].uuid(),
                                             descrs[3].uuid()]))
        self.assertEqual(len(r), 2)
        self.assertEqual(set(r),
                         {descrs[0], descrs[3]})

    def test_clear(self):
        i = MemoryDescriptorIndex()
        n = 10

        descrs = [random_descriptor() for _ in range(n)]
        i.add_many_descriptors(descrs)
        self.assertEqual(len(i), n)
        i.clear()
        self.assertEqual(len(i), 0)
        self.assertEqual(i._table, {})

    def test_has(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(10)]
        i.add_many_descriptors(descrs)

        self.assertTrue(i.has_descriptor(descrs[4].uuid()))
        self.assertFalse(i.has_descriptor('not_an_int'))

    def test_added_descriptor_table_caching(self):
        cache_elem = DataMemoryElement(readonly=False)
        descrs = [random_descriptor() for _ in range(3)]
        expected_table = dict((r.uuid(), r) for r in descrs)

        i = MemoryDescriptorIndex(cache_elem)
        self.assertTrue(cache_elem.is_empty())

        # Should add descriptors to table, caching to writable element.
        i.add_many_descriptors(descrs)
        self.assertFalse(cache_elem.is_empty())
        self.assertEqual(pickle.loads(i.cache_element.get_bytes()),
                         expected_table)

        # Changing the internal table (remove, add) it should reflect in
        # cache
        new_d = random_descriptor()
        expected_table[new_d.uuid()] = new_d
        i.add_descriptor(new_d)
        self.assertEqual(pickle.loads(i.cache_element.get_bytes()),
                         expected_table)

        rm_d = list(expected_table.values())[0]
        del expected_table[rm_d.uuid()]
        i.remove_descriptor(rm_d.uuid())
        self.assertEqual(pickle.loads(i.cache_element.get_bytes()),
                         expected_table)

    def test_remove(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        self.assertEqual(len(i), 100)
        self.assertEqual(list(i.iterdescriptors()), descrs)

        # remove singles
        i.remove_descriptor(descrs[0].uuid())
        self.assertEqual(len(i), 99)
        self.assertEqual(set(i.iterdescriptors()),
                         set(descrs[1:]))

        # remove many
        rm_d = descrs[slice(45, 80, 3)]
        i.remove_many_descriptors((d.uuid() for d in rm_d))
        self.assertEqual(len(i), 99 - len(rm_d))
        self.assertEqual(set(i.iterdescriptors()),
                         set(descrs[1:]).difference(rm_d))

    def test_iterdescrs(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        self.assertEqual(set(i.iterdescriptors()),
                         set(descrs))

    def test_iterkeys(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        self.assertEqual(set(i.iterkeys()),
                         set(d.uuid() for d in descrs))

    def test_iteritems(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        self.assertEqual(set(six.iteritems(i)),
                         set((d.uuid(), d) for d in descrs))
