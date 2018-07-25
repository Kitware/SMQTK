from __future__ import division, print_function
import unittest

import mock

from smqtk.algorithms.nn_index.lsh.functors import \
    LshFunctor, get_lsh_functor_impls


class TestLshFunctorImplGetter (unittest.TestCase):

    @mock.patch('smqtk.algorithms.nn_index.lsh.functors.plugin.get_plugins')
    def test_get_lsh_functor_impls_no_reload(self, m_get_plugins):
        get_lsh_functor_impls()
        m_get_plugins.assert_called_once()
        self.assertEqual(m_get_plugins.call_args[0][0],
                         'smqtk.algorithms.nn_index.lsh.functors')
        self.assertEqual(m_get_plugins.call_args[0][2],
                         'LSH_FUNCTOR_PATH')
        self.assertEqual(m_get_plugins.call_args[0][3],
                         'LSH_FUNCTOR_CLASS')
        self.assertEqual(m_get_plugins.call_args[0][4],
                         LshFunctor)
        self.assertFalse(m_get_plugins.call_args[1]['reload_modules'])

    @mock.patch('smqtk.algorithms.nn_index.lsh.functors.plugin.get_plugins')
    def test_get_lsh_functor_impls_with_reload(self, m_get_plugins):
        get_lsh_functor_impls(True)
        m_get_plugins.assert_called_once()
        self.assertEqual(m_get_plugins.call_args[0][0],
                         'smqtk.algorithms.nn_index.lsh.functors')
        # m_get_plugins.call_args[0][1] is a path depending on where the python
        # code is.
        self.assertEqual(m_get_plugins.call_args[0][2],
                         'LSH_FUNCTOR_PATH')
        self.assertEqual(m_get_plugins.call_args[0][3],
                         'LSH_FUNCTOR_CLASS')
        self.assertEqual(m_get_plugins.call_args[0][4],
                         LshFunctor)
        self.assertTrue(m_get_plugins.call_args[1]['reload_modules'])
