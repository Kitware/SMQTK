import cPickle
import os
import tempfile
import unittest

import nose.tools as ntools
import numpy

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex


__author__ = "paul.tunison@kitware.com"


RAND_UUID = 0


def random_descriptor():
    global RAND_UUID
    d = DescriptorMemoryElement('random', RAND_UUID)
    d.set_vector(numpy.random.rand(64))
    RAND_UUID += 1
    return d


class TestMemoryDescriptorIndex (unittest.TestCase):

    def tearDown(self):
        # clear memory descriptor cache
        DescriptorMemoryElement.MEMORY_CACHE = {}

    def test_is_usable(self):
        ntools.assert_equal(MemoryDescriptorIndex.is_usable(), True)

    def test_init_no_cache(self):
        inst = MemoryDescriptorIndex()
        ntools.assert_is_none(inst.file_cache, None)
        ntools.assert_equal(inst._table, {})

    def test_init_nonexistant_file_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)
        os.remove(tmp_cache)

        inst = MemoryDescriptorIndex(tmp_cache)
        ntools.assert_equal(inst.file_cache, tmp_cache)
        ntools.assert_equal(inst._table, {})

    def test_init_empty_file_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)

        try:
            ntools.assert_raises(EOFError, MemoryDescriptorIndex, tmp_cache)
        finally:
            os.remove(tmp_cache)

    def test_init_with_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)

        try:
            d_list = (random_descriptor(), random_descriptor(),
                      random_descriptor(), random_descriptor())
            test_cache = dict((r.uuid(), r) for r in d_list)
            with open(tmp_cache, 'w') as f:
                cPickle.dump(test_cache, f)

            inst = MemoryDescriptorIndex(tmp_cache)
            ntools.assert_equal(len(inst._table), 4)
            ntools.assert_equal(inst.file_cache, tmp_cache)
            ntools.assert_equal(inst._table, test_cache)
            ntools.assert_equal(set(inst._table.values()), set(d_list))
        finally:
            os.remove(tmp_cache)

    def test_table_caching(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)
        os.remove(tmp_cache)

        try:
            i = MemoryDescriptorIndex(tmp_cache)
            descrs = [random_descriptor() for _ in xrange(3)]
            expected_cache = dict((r.uuid(), r) for r in descrs)

            # cache should not exist yet
            ntools.assert_false(os.path.isfile(tmp_cache))

            # Should write file and should be a dictionary of 3
            # elements
            i.add_many_descriptors(descrs)
            ntools.assert_true(os.path.isfile(tmp_cache))
            with open(tmp_cache) as f:
                ntools.assert_equal(cPickle.load(f),
                                    expected_cache)

            # Changing the internal table (remove, add) it should reflect in
            # cache
            new_d = random_descriptor()
            i.add_descriptor(new_d)
            expected_cache[new_d.uuid()] = new_d
            with open(tmp_cache) as f:
                ntools.assert_equal(cPickle.load(f),
                                    expected_cache)

            rm_d = expected_cache.values()[0]
            i.remove_descriptor(rm_d.uuid())
            del expected_cache[rm_d.uuid()]
            with open(tmp_cache) as f:
                ntools.assert_equal(cPickle.load(f),
                                    expected_cache)
        finally:
            os.remove(tmp_cache)

    def test_default_config(self):
        ntools.assert_equal(
            MemoryDescriptorIndex.get_default_config(),
            {"file_cache": None, "pickle_protocol": -1}
        )

    def test_get_config(self):
        ntools.assert_equal(
            MemoryDescriptorIndex().get_config(),
            {'file_cache': None, "pickle_protocol": -1}
        )

        ntools.assert_equal(
            MemoryDescriptorIndex(None).get_config(),
            {'file_cache': None, "pickle_protocol": -1}
        )

        ntools.assert_equal(
            MemoryDescriptorIndex('/some/abs/path').get_config(),
            {'file_cache': '/some/abs/path', "pickle_protocol": -1}
        )

        ntools.assert_equal(
            MemoryDescriptorIndex('some/rel/path').get_config(),
            {'file_cache': 'some/rel/path', "pickle_protocol": -1}
        )

    def test_from_config(self):
        inst = MemoryDescriptorIndex.from_config({'file_cache': None})
        ntools.assert_is_none(inst.file_cache)

        fp = '/doesnt/exist/yet'
        inst = MemoryDescriptorIndex.from_config({'file_cache': fp})
        ntools.assert_equal(inst.file_cache, fp)

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

        d2 = random_descriptor()
        index.add_descriptor(d2)
        ntools.assert_equal(index.count(), 2)

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

        descrs = [random_descriptor() for _ in xrange(n)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(len(i), n)
        i.clear()
        ntools.assert_equal(len(i), 0)
        ntools.assert_equal(i._table, {})

    def test_has(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in xrange(10)]
        i.add_many_descriptors(descrs)

        ntools.assert_true(i.has_descriptor(descrs[4].uuid()))
        ntools.assert_false(i.has_descriptor('not_an_int'))

    def test_remove(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in xrange(100)]
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
        descrs = [random_descriptor() for _ in xrange(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(set(i.iterdescriptors()),
                            set(descrs))

    def test_iterkeys(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in xrange(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(set(i.iterkeys()),
                            set(d.uuid() for d in descrs))

    def test_iteritems(self):
        i = MemoryDescriptorIndex()
        descrs = [random_descriptor() for _ in xrange(100)]
        i.add_many_descriptors(descrs)
        ntools.assert_equal(set(i.iteritems()),
                            set((d.uuid(), d) for d in descrs))
