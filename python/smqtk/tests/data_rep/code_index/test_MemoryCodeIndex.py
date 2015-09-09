import cPickle
import os
import tempfile
import unittest

import nose.tools as ntools
import numpy

from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorMemoryElement
from smqtk.data_rep.code_index.memory import MemoryCodeIndex

__author__ = 'purg'


RAND_UUID = 0


def random_descriptor():
    global RAND_UUID
    d = DescriptorMemoryElement('random', RAND_UUID)
    d.set_vector(numpy.random.rand(64))
    RAND_UUID += 1
    return d


class TestMemoryCodeIndex (unittest.TestCase):

    def test_is_usable(self):
        ntools.assert_equal(MemoryCodeIndex.is_usable(), True)

    def test_init_no_cache(self):
        inst = MemoryCodeIndex()
        ntools.assert_equal(inst._num_descr, 0)
        ntools.assert_is_none(inst._file_cache, None)
        ntools.assert_equal(inst._table, {})

    def test_init_nonexistant_file_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)
        os.remove(tmp_cache)

        inst = MemoryCodeIndex(tmp_cache)
        ntools.assert_equal(inst._num_descr, 0)
        ntools.assert_equal(inst._file_cache, tmp_cache)
        ntools.assert_equal(inst._table, {})

    def test_init_empty_file_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)

        try:
            ntools.assert_raises(EOFError, MemoryCodeIndex, tmp_cache)
        finally:
            os.remove(tmp_cache)

    def test_init_with_cache(self):
        fd, tmp_cache = tempfile.mkstemp()
        os.close(fd)

        try:
            test_cache = {
                0: {0: 'foo'},
                5: {0: 'bar'},
                2354: {0: 'baz', 1: "fak"},
            }
            with open(tmp_cache, 'w') as f:
                cPickle.dump(test_cache, f)

            inst = MemoryCodeIndex(tmp_cache)
            ntools.assert_equal(inst._num_descr, 4)
            ntools.assert_equal(inst._file_cache, tmp_cache)
            ntools.assert_equal(inst._table, test_cache)
        finally:
            os.remove(tmp_cache)

    def test_default_config(self):
        ntools.assert_equal(
            MemoryCodeIndex.default_config(),
            {"file_cache": None}
        )

    def test_from_config(self):
        inst = MemoryCodeIndex.from_config({'file_cache': None})
        ntools.assert_is_none(inst._file_cache)

        fp = '/doesnt/exist/yet'
        inst = MemoryCodeIndex.from_config({'file_cache': fp})
        ntools.assert_equal(inst._file_cache, fp)

    def test_add_descriptor(self):
        index = MemoryCodeIndex()

        d1 = random_descriptor()
        index.add_descriptor(0, d1)
        ntools.assert_equal(index._table[0][d1.uuid()], d1)

        d2 = random_descriptor()
        index.add_descriptor(5213, d2)
        ntools.assert_equal(index._table[5213][d2.uuid()], d2)

    def test_add_many(self):
        code_descrs = [
            (0, random_descriptor()),
            (1, random_descriptor()),
            (3, random_descriptor()),
            (0, random_descriptor()),
            (8, random_descriptor()),
        ]
        index = MemoryCodeIndex()
        index.add_many_descriptors(code_descrs)

        # Compare code keys of input to code keys in internal table
        ntools.assert_equal(set(index._table.keys()),
                            set([e[0] for e in code_descrs]))

        # Get the set of descriptors in the internal table and compare it with
        # the set of generated random descriptors.
        r_set = set()
        [r_set.update(d.values()) for d in index._table.values()]
        ntools.assert_equal(
            set([e[1] for e in code_descrs]),
            r_set
        )

    def test_count(self):
        index = MemoryCodeIndex()
        ntools.assert_equal(index.count(), 0)

        d1 = random_descriptor()
        index.add_descriptor(0, d1)
        ntools.assert_equal(index.count(), 1)

        d2 = random_descriptor()
        index.add_descriptor(1, d2)
        ntools.assert_equal(index.count(), 2)

    def test_get_descriptors(self):
        code_descrs = [
            (0, random_descriptor()),  # [0]
            (1, random_descriptor()),  # [1]
            (3, random_descriptor()),  # [2]
            (0, random_descriptor()),  # [3]
            (8, random_descriptor()),  # [4]
        ]
        index = MemoryCodeIndex()
        index.add_many_descriptors(code_descrs)

        # single descriptor reference
        r = list(index.get_descriptors(1))
        ntools.assert_equal(len(r), 1)
        ntools.assert_equal(r[0], code_descrs[1][1])

        # multiple descriptor reference
        r = list(index.get_descriptors(0))
        ntools.assert_equal(len(r), 2)
        ntools.assert_equal(set(r),
                            {code_descrs[0][1], code_descrs[3][1]})

        # multiple code query
        r = list(index.get_descriptors([0, 3, 8]))
        ntools.assert_equal(len(r), 4)
        ntools.assert_equal(set(r),
                            {code_descrs[0][1], code_descrs[2][1],
                             code_descrs[3][1], code_descrs[4][1]})
