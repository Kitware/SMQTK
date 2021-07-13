from __future__ import division, print_function
import unittest

import unittest.mock as mock

from smqtk.representation import DescriptorElement
from smqtk.representation.descriptor_set import DescriptorSet


class DummyDescriptorSet (DescriptorSet):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def get_descriptor(self, uuid):
        pass

    def get_many_descriptors(self, uuids):
        pass

    def iterkeys(self):
        pass

    def iteritems(self):
        pass

    def iterdescriptors(self):
        pass

    def remove_many_descriptors(self, uuids):
        pass

    def has_descriptor(self, uuid):
        pass

    def add_many_descriptors(self, descriptors):
        pass

    def count(self):
        pass

    def clear(self):
        pass

    def remove_descriptor(self, uuid):
        pass

    def add_descriptor(self, descriptor):
        pass


class TestDescriptorSetAbstract (unittest.TestCase):

    def test_len(self):
        di = DummyDescriptorSet()
        di.count = mock.Mock(return_value=100)
        self.assertEqual(len(di), 100)
        di.count.assert_called_once_with()

    def test_get_item(self):
        di = DummyDescriptorSet()
        di.get_descriptor = mock.Mock(return_value='foo')
        self.assertEqual(di['some_key'], 'foo')
        di.get_descriptor.assert_called_once_with('some_key')

    def test_del_item(self):
        di = DummyDescriptorSet()
        di.remove_descriptor = mock.Mock()

        del di['foo']
        di.remove_descriptor.assert_called_once_with('foo')

    def test_iter(self):
        # Iterating over a DescriptorSet should yield the descriptor elements
        di = DummyDescriptorSet()

        def dumb_iterator():
            for _i in range(3):
                yield _i

        di.iterdescriptors = mock.Mock(side_effect=dumb_iterator)

        for i, v in enumerate(iter(di)):
            self.assertEqual(i, v)
        self.assertEqual(list(di), [0, 1, 2])
        self.assertEqual(tuple(di), (0, 1, 2))
        self.assertEqual(di.iterdescriptors.call_count, 3)

    @mock.patch("smqtk.representation.descriptor_set.DescriptorElement"
                ".get_many_vectors", wraps=DescriptorElement.get_many_vectors)
    def test_get_many_vectors_empty(self, m_de_gmv):
        """ Test that no vectors are returned when no UIDs are provided. """
        inst = DummyDescriptorSet()
        inst.get_many_descriptors = mock.Mock(return_value=[])
        r = inst.get_many_vectors([])
        assert r == []
        m_de_gmv.assert_called_once_with([])
