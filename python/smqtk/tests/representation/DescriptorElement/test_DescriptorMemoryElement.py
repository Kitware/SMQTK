import unittest

import numpy
from six import BytesIO
from six.moves import cPickle, cStringIO

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


class TestDescriptorMemoryElement (unittest.TestCase):

    def test_configuration(self):
        default_config = DescriptorMemoryElement.get_default_config()
        self.assertEqual(default_config, {})

        inst1 = DescriptorMemoryElement.from_config(default_config, 'test', 'a')
        self.assertEqual(default_config, inst1.get_config())
        self.assertEqual(inst1.type(), 'test')
        self.assertEqual(inst1.uuid(), 'a')

        # vector-based equality
        inst2 = DescriptorMemoryElement.from_config(inst1.get_config(),
                                                    'test', 'a')
        self.assertEqual(inst1, inst2)

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

    def test_set_state_version_1(self):
        # Test support of older state version
        expected_type = 'test-type'
        expected_uid = 'test-uid'
        expected_v = numpy.array([1, 2, 3])
        expected_v_b = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(expected_v_b, expected_v)
        expected_v_dump = expected_v_b.getvalue()

        e = DescriptorMemoryElement(None, None)
        e.__setstate__((expected_type, expected_uid, expected_v_dump))
        self.assertEqual(e.type(), expected_type)
        self.assertEqual(e.uuid(), expected_uid)
        numpy.testing.assert_array_equal(e.vector(), expected_v)

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
        self.assertTrue((v == 0).all())
        self.assertFalse(sum(t) == 0.)
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
        self.assertTrue((v1 == 0).all())
        self.assertTrue((v2 == 0).all())
        self.assertTrue((v3 == 0).all())
        self.assertFalse(sum(t1) == 0.)
        self.assertFalse(sum(t2) == 0.)
        self.assertFalse(sum(t3) == 0.)
        numpy.testing.assert_equal(d1.vector(), t1)
        numpy.testing.assert_equal(d2.vector(), t2)
        numpy.testing.assert_equal(d3.vector(), t3)

    def test_output_immutability(self):
        # make sure that data stored is not susceptible to modifications after
        # extraction
        v = numpy.ones(16)
        d = DescriptorMemoryElement('test', 0)
        self.assertFalse(d.has_vector())
        d.set_vector(v)
        r = d.vector()
        r[:] = 0
        self.assertEqual(r.sum(), 0)
        self.assertEqual(d.vector().sum(), 16)

    def test_none_set(self):
        d = DescriptorMemoryElement('test', 0)
        self.assertFalse(d.has_vector())

        d.set_vector(numpy.ones(16))
        self.assertTrue(d.has_vector())
        numpy.testing.assert_equal(d.vector(), numpy.ones(16))

        d.set_vector(None)
        self.assertFalse(d.has_vector())
        self.assertIs(d.vector(), None)
