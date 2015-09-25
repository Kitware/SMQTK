import unittest

import mock
import nose.tools as ntools
import numpy

from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
from smqtk.representation.code_index import CodeIndex

__author__ = "paul.tunison@kitware.com"


RAND_UUID = 0


def random_descriptor():
    global RAND_UUID
    d = DescriptorMemoryElement('random', RAND_UUID)
    d.set_vector(numpy.random.rand(64))
    RAND_UUID += 1
    return d


class DummyCodeIndex (CodeIndex):

    @classmethod
    def is_usable(cls):
        return True

    add_descriptor = mock.Mock()
    add_many_descriptors = mock.Mock()
    codes = mock.Mock()
    count = mock.Mock()
    clear = mock.Mock()
    get_config = mock.Mock()
    get_descriptors = mock.Mock()


class TestCodeIndexAbstract (unittest.TestCase):

    @mock.patch.object(DummyCodeIndex, 'count')
    def test_len(self, mock_count):
        mock_count.return_value = 100
        index = DummyCodeIndex()
        ntools.assert_equal(len(index), 100)
        DummyCodeIndex.count.assert_called_once_with()

    @mock.patch.object(DummyCodeIndex, 'add_descriptor')
    def test_setitem(self, mock_add_descr):
        index = DummyCodeIndex()

        d1 = random_descriptor()
        index[12345] = d1
        mock_add_descr.assert_called_once_with(12345, d1)

    @mock.patch.object(DummyCodeIndex, 'get_descriptors')
    def test_getitem(self, mock_get_descr):
        index = DummyCodeIndex()

        _ = index[0]
        mock_get_descr.assert_called_once_with(0)

        _ = index[0, 5, 123]
        mock_get_descr.assert_called_with((0, 5, 123))

        _ = index[[0, 5, 123]]
        mock_get_descr.assert_called_with([0, 5, 123])
