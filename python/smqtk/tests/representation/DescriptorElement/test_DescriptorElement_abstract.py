from __future__ import division, print_function
import mock
import numpy
import unittest

from smqtk.representation import DescriptorElement


class DummyDescriptorElement (DescriptorElement):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        """ stub """

    def set_vector(self, new_vec):
        """ stub """

    def has_vector(self):
        """ stub """

    def vector(self):
        """ stub """


class TestDescriptorElementAbstract (unittest.TestCase):

    def test_init(self):
        expected_uuid = 'some uuid'
        expected_type_str = 'some type'
        de = DummyDescriptorElement(expected_type_str, expected_uuid)
        self.assertEqual(de.type(), expected_type_str)
        self.assertEqual(de.uuid(), expected_uuid)

    def test_equality(self):
        de1 = DummyDescriptorElement('t', 'u1')
        de2 = DummyDescriptorElement('t', 'u2')
        de1.vector = de2.vector = \
            mock.Mock(return_value=numpy.random.randint(0, 10, 10))

        self.assertTrue(de1 == de1)
        self.assertTrue(de2 == de2)
        self.assertTrue(de1 == de2)
        self.assertFalse(de1 != de2)

    def test_equality_diffTypeStr(self):
        # Type strings no longer considered in descriptor equality nor hashing.
        v = numpy.random.randint(0, 10, 10)
        d1 = DummyDescriptorElement('a', 'u')
        d2 = DummyDescriptorElement('b', 'u')
        d1.vector = d2.vector = mock.Mock(return_value=v)
        self.assertFalse(d1 != d2)
        self.assertTrue(d1 == d2)

    def test_nonEquality_diffInstance(self):
        # diff instance
        de = DummyDescriptorElement('a', 'b')
        self.assertFalse(de == 'string')
        self.assertTrue(de != 'string')

    def test_nonEquality_diffVectors(self):
        # different vectors (same size)
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 10)

        d1 = DummyDescriptorElement('a', 'b')
        d1.vector = mock.Mock(return_value=v1)

        d2 = DummyDescriptorElement('a', 'b')
        d2.vector = mock.Mock(return_value=v2)

        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)

    def test_nonEquality_diffVectorSize(self):
        # different sized vectors
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 100)

        d1 = DummyDescriptorElement('a', 'b')
        d1.vector = mock.Mock(return_value=v1)

        d2 = DummyDescriptorElement('a', 'b')
        d2.vector = mock.Mock(return_value=v2)

        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)

    def test_hash(self):
        # Hash of a descriptor element is solely based on the UUID value of that
        # element.
        t1 = 'a'
        uuid1 = 'some uuid'
        de1 = DummyDescriptorElement(t1, uuid1)

        t2 = 'b'
        uuid2 = 'some uuid'
        de2 = DummyDescriptorElement(t2, uuid2)

        self.assertEqual(hash(de1), hash(uuid1))
        self.assertEqual(hash(de2), hash(uuid2))
        self.assertEqual(hash(de1), hash(de2))

    def test_getstate(self):
        expected_type = 'a'
        expected_uid = 'b'
        e = DummyDescriptorElement(expected_type, expected_uid)
        state = e.__getstate__()
        self.assertDictEqual(
            state,
            {
                '_type_label': expected_type,
                '_uuid': expected_uid,
            }
        )

    def test_setstate(self):
        # Intentionally bad input types.
        # noinspection PyTypeChecker
        e = DummyDescriptorElement(None, None)
        self.assertIsNone(e._type_label)
        self.assertIsNone(e._uuid)

        expected_type = 'a'
        expected_uid = 'b'
        state = {
            '_type_label': expected_type,
            '_uuid': expected_uid,
        }
        e.__setstate__(state)
        self.assertEqual(e._type_label, expected_type)
        self.assertEqual(e._uuid, expected_uid)
