import unittest

import mock

from smqtk.algorithms.nn_index.lsh.functors import LshFunctor


class DummyLshFunctor (LshFunctor):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def get_hash(self, descriptor):
        pass


class testLshFunctorAbstract (unittest.TestCase):

    def test_call(self):
        # calling an instance should get us to the get_hash method.
        f = DummyLshFunctor()
        f.get_hash = mock.MagicMock()

        expected_descriptor = 'pretend descriptor element'
        f(expected_descriptor)
        f.get_hash.assert_called_once_with(expected_descriptor)
