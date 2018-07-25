from __future__ import division, print_function
import abc
import os
import unittest

from smqtk.utils.plugin import Pluggable, get_plugins, OS_ENV_PATH_SEP


class DummyInterface (Pluggable):

    @abc.abstractmethod
    def inst_method(self, val):
        """
        dummy abstract function
        """


class TestGetPluginGeneric (unittest.TestCase):

    INTERNAL_PLUGIN_DIR = os.path.join(os.path.dirname(__file__),
                                       "test_plugin_dir",
                                       "internal_plugins")

    INTERNAL_PLUGIN_MOD_PATH = \
        'smqtk.tests.utils.test_plugin_dir.internal_plugins'

    ENV_VAR = "TEST_PLUGIN_MODULE_PATH"
    HELP_VAR = "TEST_PLUGIN_CLASS"

    EXT_MOD_1 = 'smqtk.tests.utils.test_plugin_dir.external_1'
    EXT_MOD_2 = 'smqtk.tests.utils.test_plugin_dir.external_2'

    @classmethod
    def get_dummy_plugins(cls):
        return get_plugins(cls.INTERNAL_PLUGIN_MOD_PATH,
                           cls.INTERNAL_PLUGIN_DIR, cls.ENV_VAR,
                           cls.HELP_VAR, DummyInterface)

    def test_get_internal_modules(self, do_return=False):
        m = self.get_dummy_plugins()
        self.assertIn('ImplFoo', m)
        self.assertIn('ImplBar', m)
        self.assertIn('ImplDoExport', m)

        self.assertEqual(m['ImplFoo']().inst_method('a'), 'fooa')
        self.assertEqual(m['ImplBar']().inst_method('b'), 'barb')
        self.assertEqual(m['ImplDoExport']().inst_method('c'), 'doExportc')

        self.assertNotIn('ImplNotUsable', m)
        self.assertNotIn('SomethingElse', m)
        self.assertNotIn('ImplNoExport', m)
        self.assertNotIn('ImplSkipModule', m)

        if do_return:
            return m

    def test_external_1_only(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = self.EXT_MOD_1
        m = self.test_get_internal_modules(True)

        self.assertIn('ImplExternal1', m)
        self.assertIn('ImplExternal2', m)
        self.assertNotIn('ImplExternal3', m)

        self.assertEqual(m['ImplExternal1']().inst_method('d'), 'external1d')
        self.assertEqual(m['ImplExternal2']().inst_method('e'), 'external2e')

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]

    def test_external_1_with_trailing_sep(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = self.EXT_MOD_1+OS_ENV_PATH_SEP
        m = self.test_get_internal_modules(True)

        self.assertIn('ImplExternal1', m)
        self.assertIn('ImplExternal2', m)
        self.assertNotIn('ImplExternal3', m)

        self.assertEqual(m['ImplExternal1']().inst_method('d'), 'external1d')
        self.assertEqual(m['ImplExternal2']().inst_method('e'), 'external2e')

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]

    def test_external_1_with_leading_sep(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = OS_ENV_PATH_SEP+self.EXT_MOD_1
        m = self.test_get_internal_modules(True)

        self.assertIn('ImplExternal1', m)
        self.assertIn('ImplExternal2', m)
        self.assertNotIn('ImplExternal3', m)

        self.assertEqual(m['ImplExternal1']().inst_method('d'), 'external1d')
        self.assertEqual(m['ImplExternal2']().inst_method('e'), 'external2e')

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]

    def test_external_2_only(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = self.EXT_MOD_2
        m = self.test_get_internal_modules(True)

        self.assertNotIn('ImplExternal1', m)
        self.assertNotIn('ImplExternal2', m)
        self.assertIn('ImplExternal3', m)

        self.assertEqual(m['ImplExternal3']().inst_method('f'), 'external3f')

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]

    def test_external_1_and_2(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = OS_ENV_PATH_SEP.join([self.EXT_MOD_1,
                                                         self.EXT_MOD_2])
        m = self.test_get_internal_modules(True)

        self.assertIn('ImplExternal1', m)
        self.assertIn('ImplExternal2', m)
        self.assertIn('ImplExternal3', m)

        self.assertEqual(m['ImplExternal1']().inst_method('d'), 'external1d')
        self.assertEqual(m['ImplExternal2']().inst_method('e'), 'external2e')
        self.assertEqual(m['ImplExternal3']().inst_method('f'), 'external3f')

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]

    def test_junk_external_mod(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = "This is a junk string"
        # This should skip module it can't find. In this case, the junk string
        # is treated as a python module path, which is invalid. A warning is
        # emitted but the plugin query succeeds as if the invalid chunk didn't
        # exist.
        self.test_get_internal_modules()

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]

    def test_external_1_and_2_and_garbage(self):
        env_orig_value = os.environ.get(self.ENV_VAR, None)

        os.environ[self.ENV_VAR] = OS_ENV_PATH_SEP.join([self.EXT_MOD_1,
                                                         self.EXT_MOD_2,
                                                         'asdgasfhsadf',
                                                         'some thing weird',
                                                         'but still uses sep'])
        m = self.test_get_internal_modules(True)

        self.assertIn('ImplExternal1', m)
        self.assertIn('ImplExternal2', m)
        self.assertIn('ImplExternal3', m)

        self.assertEqual(m['ImplExternal1']().inst_method('d'), 'external1d')
        self.assertEqual(m['ImplExternal2']().inst_method('e'), 'external2e')
        self.assertEqual(m['ImplExternal3']().inst_method('f'), 'external3f')

        if env_orig_value:
            os.environ[self.ENV_VAR] = env_orig_value
        else:
            del os.environ[self.ENV_VAR]
