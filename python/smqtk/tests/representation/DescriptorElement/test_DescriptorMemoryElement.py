import cPickle
import unittest

import nose.tools as ntools
import numpy

from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement


__author__ = "paul.tunison@kitware.com"


class TestDescriptorMemoryElement (unittest.TestCase):

    def test_configuration(self):
        default_config = DescriptorMemoryElement.get_default_config()
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
        # Make a couple descriptors
        v1 = numpy.array([1, 2, 3])
        d1 = DescriptorMemoryElement('test', 0)
        d1.set_vector(v1)

        v2 = numpy.array([4, 5, 6])
        d2 = DescriptorMemoryElement('test', 1)
        d2.set_vector(v2)

        d1_s = cPickle.dumps(d1)
        d2_s = cPickle.dumps(d2)

        # Attempt reconstitution
        d1_r = cPickle.loads(d1_s)
        d2_r = cPickle.loads(d2_s)

        numpy.testing.assert_array_equal(v1, d1_r.vector())
        numpy.testing.assert_array_equal(v2, d2_r.vector())

    def test_input_immutability(self):
        # make sure that data stored is not susceptible to shifts in the
        # originating data matrix they were pulled from.

        #
        # Testing this with a single vector
        #
        v = numpy.random.rand(16)
        t = tuple(v.copy())
        d = DescriptorMemoryElement('test', 0)
        d.set_vector(v)
        v[:] = 0
        ntools.assert_true((v == 0).all())
        ntools.assert_false(sum(t) == 0.)
        numpy.testing.assert_equal(d.vector(), t)

        #
        # Testing with matrix
        #
        m = numpy.random.rand(20, 16)

        v1 = m[3]
        v2 = m[15]
        v3 = m[19]

        # Save truth values of arrays as immutable tuples (copies)
        t1 = tuple(v1.copy())
        t2 = tuple(v2.copy())
        t3 = tuple(v3.copy())

        d1 = DescriptorMemoryElement('test', 1)
        d1.set_vector(v1)
        d2 = DescriptorMemoryElement('test', 2)
        d2.set_vector(v2)
        d3 = DescriptorMemoryElement('test', 3)
        d3.set_vector(v3)

        numpy.testing.assert_equal(v1, d1.vector())
        numpy.testing.assert_equal(v2, d2.vector())
        numpy.testing.assert_equal(v3, d3.vector())

        # Changing the source should not change stored vectors
        m[:, :] = 0.
        ntools.assert_true((v1 == 0).all())
        ntools.assert_true((v2 == 0).all())
        ntools.assert_true((v3 == 0).all())
        ntools.assert_false(sum(t1) == 0.)
        ntools.assert_false(sum(t2) == 0.)
        ntools.assert_false(sum(t3) == 0.)
        numpy.testing.assert_equal(d1.vector(), t1)
        numpy.testing.assert_equal(d2.vector(), t2)
        numpy.testing.assert_equal(d3.vector(), t3)

    def test_output_immutability(self):
        # make sure that data stored is not susceptible to modifications after
        # extraction
        v = numpy.ones(16)
        d = DescriptorMemoryElement('test', 0)
        ntools.assert_false(d.has_vector())
        d.set_vector(v)
        r = d.vector()
        r[:] = 0
        ntools.assert_equal(r.sum(), 0)
        ntools.assert_equal(d.vector().sum(), 16)

    def test_none_set(self):
        d = DescriptorMemoryElement('test', 0)
        ntools.assert_false(d.has_vector())

        d.set_vector(numpy.ones(16))
        ntools.assert_true(d.has_vector())
        numpy.testing.assert_equal(d.vector(), numpy.ones(16))

        d.set_vector(None)
        ntools.assert_false(d.has_vector())
        ntools.assert_is(d.vector(), None)
