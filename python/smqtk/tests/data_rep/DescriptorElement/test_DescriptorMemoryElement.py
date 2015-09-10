import cPickle
import unittest

import nose.tools as ntools
import numpy

from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorMemoryElement


__author__ = 'purg'


class TestDescriptorMemoryElement (unittest.TestCase):

    def test_configuration(self):
        default_config = DescriptorMemoryElement.default_config()
        ntools.assert_equal(default_config, {})

        inst1 = DescriptorMemoryElement.from_config(default_config, 'test', 'a')
        ntools.assert_equal(default_config, inst1.get_config())
        ntools.assert_equal(inst1.type(), 'test')
        ntools.assert_equal(inst1.uuid(), 'a')

        # vector-based equality
        inst2 = DescriptorMemoryElement.from_config(inst1.get_config(),
                                                    'test', 'a')
        ntools.assert_equal(inst1, inst2)

    def test_pickle_dump_load(self):
        # Wipe current cache
        DescriptorMemoryElement.MEMORY_CACHE = {}

        # Make a couple descriptors
        v1 = numpy.array([1, 2, 3])
        d1 = DescriptorMemoryElement('test', 0)
        d1.set_vector(v1)

        v2 = numpy.array([4, 5, 6])
        d2 = DescriptorMemoryElement('test', 1)
        d2.set_vector(v2)

        ntools.assert_in(('test', 0), DescriptorMemoryElement.MEMORY_CACHE)
        ntools.assert_in(('test', 1), DescriptorMemoryElement.MEMORY_CACHE)

        d1_s = cPickle.dumps(d1)
        d2_s = cPickle.dumps(d2)

        # Wipe cache again
        DescriptorMemoryElement.MEMORY_CACHE = {}
        ntools.assert_not_in(('test', 0), DescriptorMemoryElement.MEMORY_CACHE)
        ntools.assert_not_in(('test', 1), DescriptorMemoryElement.MEMORY_CACHE)

        # Attempt reconstitution
        d1_r = cPickle.loads(d1_s)
        d2_r = cPickle.loads(d2_s)

        numpy.testing.assert_array_equal(v1, d1_r.vector())
        numpy.testing.assert_array_equal(v2, d2_r.vector())

        # Cache should now have those entries back in it
        ntools.assert_in(('test', 0), DescriptorMemoryElement.MEMORY_CACHE)
        ntools.assert_in(('test', 1), DescriptorMemoryElement.MEMORY_CACHE)
