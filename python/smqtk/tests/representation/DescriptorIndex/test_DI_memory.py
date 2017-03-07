import os
import tempfile
import unittest

import nose.tools as ntools
import numpy

from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex
from smqtk.utils import merge_dict

try:
    import cPickle as pickle
except ImportError:
    import pickle


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
        ntools.assert_equal(MemoryDescriptorIndex.is_usable(), True)

    def test_default_config(self):
        # Default should be valid for constructing a new instance.
        c = MemoryDescriptorIndex.get_default_config()
        ntools.assert_equal(MemoryDescriptorIndex.from_config(c).get_config(),
                            c)

    def test_from_config_null_cache_elem(self):
        inst = MemoryDescriptorIndex.from_config({'cache_element': None})
        ntools.assert_is_none(inst.cache_element)
        ntools.assert_equal(inst._table, {})

        inst = MemoryDescriptorIndex.from_config({
            'cache_element': {
                'type': None
            }
        })
        ntools.assert_is_none(inst.cache_element)
        ntools.assert_equal(inst._table, {})

    def test_from_config_null_cache_elem_type(self):
        # An empty cache should not trigger loading on construction.
        expected_empty_cache = DataMemoryElement()
        inst = MemoryDescriptorIndex.from_config({
            'cache_element': {
                'type': 'DataMemoryElement',
                'DataMemoryElement': {'bytes': ''}
            }
        })
        ntools.assert_equal(inst.cache_element, expected_empty_cache)
        ntools.assert_equal(inst._table, {})

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
        ntools.assert_equal(inst.cache_element, expected_cache)
        ntools.assert_equal(inst._table, expected_table)

    def test_init_no_cache(self):
        inst = MemoryDescriptorIndex()
        ntools.assert_is_none(inst.cache_element, None)
        ntools.assert_equal(inst._table, {})

    def test_init_empty_cache(self):
        cache_elem = DataMemoryElement()
        inst = MemoryDescriptorIndex(cache_element=cache_elem)
        ntools.assert_equal(inst.cache_element, cache_elem)
        ntools.assert_equal(inst._table, {})

    def test_init_with_cache(self):
        d_list = (random_descriptor(), random_descriptor(),
                  random_descriptor(), random_descriptor())
        expected_table = dict((r.uuid(), r) for r in d_list)
        expected_cache = DataMemoryElement(bytes=pickle.dumps(expected_table))

        inst = MemoryDescriptorIndex(expected_cache)
        ntools.assert_equal(len(inst._table), 4)
        ntools.assert_equal(inst.cache_element, expected_cache)
        ntools.assert_equal(inst._table, expected_table)
        ntools.assert_equal(set(inst._table.values()), set(d_list))

    def test_get_config(self):
        ntools.assert_equal(
            MemoryDescriptorIndex().get_config(),
            MemoryDescriptorIndex.get_default_config()
        )

        ntools.assert_equal(
            MemoryDescriptorIndex(None).get_config(),
            MemoryDescriptorIndex.get_default_config()
        )

        empty_elem = DataMemoryElement()
        ntools.assert_equal(
            MemoryDescriptorIndex(empty_elem).get_config(),
            merge_dict(MemoryDescriptorIndex.get_default_config(), {
                'cache_element': {'type': 'DataMemoryElement'}
            })
        )

        dict_pickle_bytes = pickle.dumps({1: 1, 2: 2, 3: 3}, -1)
        cache_elem = DataMemoryElement(bytes=dict_pickle_bytes)
        ntools.assert_equal(
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
        ntools.assert_is_none(inst.cache_element)

    def test_cache_table_empty_table(self):
        inst = MemoryDescriptorIndex(DataMemoryElement(), -1)
        inst._table = {}
        expected_table_pickle_bytes = pickle.dumps(inst._table, -1)

        inst.cache_table()
        ntools.assert_is_not_none(inst.cache_element)
        ntools.assert_equal(inst.cache_element.get_bytes(),
                            expected_table_pickle_bytes)

    def test_add_descriptor(self):
        index = MemoryDescriptorIndex()

        d1 = random_descriptor()
        index.add_descriptor(d1)
        ntools.assert_equal(index._table[d1.uuid()], d1)

        d2 = random_descriptor()
        index.add_descriptor(d2)
        ntools.assert_equal(index._table[d2.uuid()], d2)

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
        ntools.assert_equal(set(index._table.keys()),
                            set([e.uuid() for e in descrs]))

        # Get the set of descriptors in the internal table and compare it with
        # the set of generated random descriptors.
        r_set = set()
        [r_set.add(d) for d in index._table.values()]
        ntools.assert_equal(
            set([e for e in descrs]),
            r_set
        )

    def test_count(self):
        index = MemoryDescriptorIndex()
        ntools.assert_equal(index.count(), 0)

        d1 = random_descriptor()
        index.add_descriptor(d1)
        ntools.assert_equal(index.count(), 1)

        d2, d3, d4 = random_descriptor(), random_descriptor(), random_descriptor()
        index.add_many_descriptors([d2, d3, d4])
        ntools.assert_equal(index.count(), 4)

        d5 = random_descriptor()
        index.add_descriptor(d5)
        ntools.assert_equal(index.count(), 5)

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
        ntools.assert_equal(r, descrs[1])

        # multiple descriptor reference
        r = list(index.get_many_descriptors([descrs[0].uuid(),
                                             descrs[3].uuid()]))
        ntools.assert_equal(len(r), 2)
        ntools.assert_equal(set(r),
                            {descrs[0], descrs[3]})

    def test_clear(self):
        i = MemoryDescriptorIndex()
        n = 10

        descrs = [random_descriptor() for _ in range(n)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(len(i), n)
        i.clear()
        ntools.assert_equal(len(i), 0)
        ntools.assert_equal(i._table, {})

    def test_has(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(10)]
        i.add_many_descriptors(descrs)

        ntools.assert_true(i.has_descriptor(descrs[4].uuid()))
        ntools.assert_false(i.has_descriptor('not_an_int'))

    def test_added_descriptor_table_caching(self):
        cache_elem = DataMemoryElement(readonly=False)
        descrs = [random_descriptor() for _ in range(3)]
        expected_table = dict((r.uuid(), r) for r in descrs)

        i = MemoryDescriptorIndex(cache_elem)
        ntools.assert_true(cache_elem.is_empty())

        # Should add descriptors to table, caching to writable element.
        i.add_many_descriptors(descrs)
        ntools.assert_false(cache_elem.is_empty())
        ntools.assert_equal(pickle.loads(i.cache_element.get_bytes()),
                            expected_table)

        # Changing the internal table (remove, add) it should reflect in
        # cache
        new_d = random_descriptor()
        expected_table[new_d.uuid()] = new_d
        i.add_descriptor(new_d)
        ntools.assert_equal(pickle.loads(i.cache_element.get_bytes()),
                            expected_table)

        rm_d = expected_table.values()[0]
        del expected_table[rm_d.uuid()]
        i.remove_descriptor(rm_d.uuid())
        ntools.assert_equal(pickle.loads(i.cache_element.get_bytes()),
                            expected_table)

    def test_remove(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(len(i), 100)
        ntools.assert_equal(list(i.iterdescriptors()), descrs)

        # remove singles
        i.remove_descriptor(descrs[0].uuid())
        ntools.assert_equal(len(i), 99)
        ntools.assert_equal(set(i.iterdescriptors()),
                            set(descrs[1:]))

        # remove many
        rm_d = descrs[slice(45, 80, 3)]
        i.remove_many_descriptors((d.uuid() for d in rm_d))
        ntools.assert_equal(len(i), 99 - len(rm_d))
        ntools.assert_equal(set(i.iterdescriptors()),
                            set(descrs[1:]).difference(rm_d))

    def test_iterdescrs(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(set(i.iterdescriptors()),
                            set(descrs))

    def test_iterkeys(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(set(i.iterkeys()),
                            set(d.uuid() for d in descrs))

    def test_iteritems(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in range(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(set(i.iteritems()),
                            set((d.uuid(), d) for d in descrs))
