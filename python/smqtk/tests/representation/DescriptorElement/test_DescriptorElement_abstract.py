import mock
import nose.tools as ntools
import numpy
import unittest

from smqtk.representation import DescriptorElement


__author__ = "paul.tunison@kitware.com"


class DummyDescriptorElement (DescriptorElement):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def set_vector(self, new_vec):
        pass

    def has_vector(self):
        pass

    def vector(self):
        pass


class TestDescriptorElementAbstract (unittest.TestCase):

    def test_init(self):
        expected_uuid = 'some uuid'
        expected_type_str = 'some type'
        de = DummyDescriptorElement(expected_type_str, expected_uuid)
        ntools.assert_equal(de.type(), expected_type_str)
        ntools.assert_equal(de.uuid(), expected_uuid)

    def test_equality(self):
        de1 = DummyDescriptorElement('t', 'u1')
        de2 = DummyDescriptorElement('t', 'u2')
        de1.vector = de2.vector = \
            mock.Mock(return_value=numpy.random.randint(0, 10, 10))

        ntools.assert_true(de1 == de1)
        ntools.assert_true(de2 == de2)
        ntools.assert_true(de1 == de2)
        ntools.assert_false(de1 != de2)

    def test_nonEquality_diffInstance(self):
        # diff instance
        de = DummyDescriptorElement('a', 'b')
        ntools.assert_false(de == 'string')
        ntools.assert_true(de != 'string')

    def test_nonEquality_diffVectors(self):
        # different vectors (same size)
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 10)

        d1 = DummyDescriptorElement('a', 'b')
        d1.vector = mock.Mock(return_value=v1)

        d2 = DummyDescriptorElement('a', 'b')
        d2.vector = mock.Mock(return_value=v2)

        ntools.assert_false(d1 == d2)
        ntools.assert_true(d1 != d2)

    def test_nonEquality_diffVectorSize(self):
        # different sized vectors
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 100)

        d1 = DummyDescriptorElement('a', 'b')
        d1.vector = mock.Mock(return_value=v1)

        d2 = DummyDescriptorElement('a', 'b')
        d2.vector = mock.Mock(return_value=v2)

        ntools.assert_false(d1 == d2)
        ntools.assert_true(d1 != d2)

    def test_nonEquality_diffTypeStr(self):
        v = numpy.random.randint(0, 10, 10)
        d1 = DummyDescriptorElement('a', 'u')
        d2 = DummyDescriptorElement('b', 'u')
        d1.vector = d2.vector = mock.Mock(return_value=v)
        ntools.assert_false(d1 == d2)
        ntools.assert_true(d1 != d2)

    def test_hash(self):
        t = 'a'
        uuid = 'some uuid'
        de = DummyDescriptorElement(t, uuid)
        ntools.assert_equal(hash(de), hash((t, uuid)))
