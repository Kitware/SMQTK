import cPickle
import os
import tempfile
import unittest

import nose.tools as ntools
import numpy

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import DescriptorMemoryIndex


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
        ntools.assert_equal(DescriptorMemoryIndex.is_usable(), True)

    def test_init_no_cache(self):
        inst = DescriptorMemoryIndex()
        ntools.assert_is_none(inst._file_cache, None)
        ntools.assert_equal(inst._table, {})

    def test_init_nonexistant_file_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)
        os.remove(tmp_cache)

        inst = DescriptorMemoryIndex(tmp_cache)
        ntools.assert_equal(inst._file_cache, tmp_cache)
        ntools.assert_equal(inst._table, {})

    def test_init_empty_file_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)

        try:
            ntools.assert_raises(EOFError, DescriptorMemoryIndex, tmp_cache)
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

            inst = DescriptorMemoryIndex(tmp_cache)
            ntools.assert_equal(len(inst._table), 4)
            ntools.assert_equal(inst._file_cache, tmp_cache)
            ntools.assert_equal(inst._table, test_cache)
            ntools.assert_equal(set(inst._table.values()), set(d_list))
        finally:
            os.remove(tmp_cache)

    def test_default_config(self):
        ntools.assert_equal(
            DescriptorMemoryIndex.get_default_config(),
            {"file_cache": None}
        )

    def test_from_config(self):
        inst = DescriptorMemoryIndex.from_config({'file_cache': None})
        ntools.assert_is_none(inst._file_cache)

        fp = '/doesnt/exist/yet'
        inst = DescriptorMemoryIndex.from_config({'file_cache': fp})
        ntools.assert_equal(inst._file_cache, fp)

    def test_add_descriptor(self):
        index = DescriptorMemoryIndex()

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
        index = DescriptorMemoryIndex()
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
        index = DescriptorMemoryIndex()
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
        index = DescriptorMemoryIndex()
        index.add_many_descriptors(descrs)

        # single descriptor reference
        r = index.get_descriptor(descrs[1].uuid())
        ntools.assert_equal(r, descrs[1])

        # multiple descriptor reference
        r = list(index.get_many_descriptors(descrs[0].uuid(),
                                            descrs[3].uuid()))
        ntools.assert_equal(len(r), 2)
        ntools.assert_equal(set(r),
                            {descrs[0], descrs[3]})
