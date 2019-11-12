from __future__ import division, print_function

import os
import pkg_resources

import pytest
# noinspection PyUnresolvedReferences
from six.moves import mock  # move defined in ``smqtk.tests``

# noinspection PyProtectedMember
from smqtk.utils.plugin import (
    EXTENSION_NAMESPACE,
    OS_ENV_PATH_SEP,
    _get_extension_plugin_modules,
    get_plugins,
)

from .test_plugin_dir.internal_plugins.interface import DummyInterface


###############################################################################
# Test constants

ENV_VAR = "TEST_PLUGIN_MODULE_PATH"
HELP_VAR = "TEST_PLUGIN_CLASS"

EXT_MOD_1 = 'tests.utils.test_plugin_dir.external_1'
EXT_MOD_2 = 'tests.utils.test_plugin_dir.external_2'
EXT_MOD_LIST = 'tests.utils.test_plugin_dir.external_list'
EXT_MOD_INVALID = 'tests.utils.test_plugin_dir.external_invalid_helper'
EXT_MOD_ABSTRACT = 'tests.utils.test_plugin_dir.external_still_abstract'


def get_plugins_for_class(cls, warn=False):
    """
    Test standard wrapper on get_plugins call using test constants.
    This is not a fixture due to environment variable mocking.
    """
    # Suppressing warnings for testing purposes.
    return get_plugins(cls, ENV_VAR, HELP_VAR, warn=warn)


###############################################################################
# Tests

def test_get_plugins_invalid_baseclass():
    """
    Baseclass provided must inherit from Pluggable.
    """
    with pytest.raises(ValueError) as execinfo:
        get_plugins_for_class(object)
    assert "Required base-class must descend from the Pluggable interface!" \
           in str(execinfo.value)


def test_get_internal_modules():
    """
    Test that known subclasses, defined in sibling modules to the module that
    defines the interface, are discovered. Testing based on known class names
    so as not to potentially polute the test by importing the specific
    subclasses (we're supposed to be able to discover them without knowing where
    they are).
    """
    class_set = get_plugins_for_class(DummyInterface)
    class_dict = {t.__name__: t for t in class_set}
    assert len(class_set) == 3

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    # Classes we do not expect to be discovered
    assert 'ImplNotUsable' not in class_dict
    assert 'SomethingElse' not in class_dict
    assert 'ImplNoExport' not in class_dict
    assert 'ImplSkipModule' not in class_dict
    assert 'ImplActuallyValid' not in class_dict

    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_1})
def test_external_1_only():
    """
    Test that we also pick up now subclasses in the external_1 module based
    on environment variable pickup.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 5

    class_dict = {t.__name__: t for t in class_set}

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal1' in class_dict
    assert 'ImplExternal2' in class_dict
    # Not expected to be picked up from external_2 module
    assert 'ImplExternal3' not in class_dict

    # Check that new classes function as expected
    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'
    assert class_dict['ImplExternal1']().inst_method('d') == "external1d"
    assert class_dict['ImplExternal2']().inst_method('e') == "external2e"


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_1+OS_ENV_PATH_SEP})
def test_external_1_with_trailing_sep():
    """
    Test that a trailing PATH separator does not impact subclass discovery.
    Result should be the same as if the trailing separator was not there.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 5

    class_dict = {t.__name__: t for t in class_set}

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal1' in class_dict
    assert 'ImplExternal2' in class_dict
    # Not expected to be picked up from external_2 module
    assert 'ImplExternal3' not in class_dict

    # Check that new classes function as expected
    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'
    assert class_dict['ImplExternal1']().inst_method('d') == "external1d"
    assert class_dict['ImplExternal2']().inst_method('e') == "external2e"


@mock.patch.dict(os.environ, {ENV_VAR: OS_ENV_PATH_SEP+EXT_MOD_1})
def test_external_1_with_leading_sep():
    """
    Test that a leading PATH separator does not impact subclass discovery.
    Result should be the same as if the leading separator was not there.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 5

    class_dict = {t.__name__: t for t in class_set}

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal1' in class_dict
    assert 'ImplExternal2' in class_dict
    # Not expected to be picked up from external_2 module
    assert 'ImplExternal3' not in class_dict

    # Check that new classes function as expected
    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'
    assert class_dict['ImplExternal1']().inst_method('d') == "external1d"
    assert class_dict['ImplExternal2']().inst_method('e') == "external2e"


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_2})
def test_external_2_only():
    """
    Test loading only external_2 module for additional subclasses.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 4

    class_dict = {t.__name__: t for t in class_set}

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal3' in class_dict
    # Not expected to be picked up from external_2 module
    assert 'ImplExternal1' not in class_dict
    assert 'ImplExternal2' not in class_dict

    # Check that new classes function as expected
    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'
    assert class_dict['ImplExternal3']().inst_method('d') == "external3d"


@mock.patch.dict(os.environ, {ENV_VAR: OS_ENV_PATH_SEP.join([EXT_MOD_1,
                                                             EXT_MOD_2])})
def test_external_1_and_2():
    """
    Test loading both external_1 and external_2 module subclasses.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 6

    class_dict = {t.__name__: t for t in class_set}

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal1' in class_dict
    assert 'ImplExternal2' in class_dict
    assert 'ImplExternal3' in class_dict

    # Check that new classes function as expected
    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'
    assert class_dict['ImplExternal1']().inst_method('d') == "external1d"
    assert class_dict['ImplExternal2']().inst_method('e') == "external2e"
    assert class_dict['ImplExternal3']().inst_method('f') == "external3f"


@mock.patch.dict(os.environ, {ENV_VAR: "This is a junk string"})
def test_junk_external_mod():
    """
    Test that invalid values to the environment parameter does not break
    functionality.

    We should skip modules that cannot be found. In this case, the junk string
    is treated as a python module path, which is invalid. A warning is emitted
    but the plugin query succeeds as if the invalid chunk didn't exist.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 3

    class_dict = {t.__name__: t for t in class_set}

    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict

    assert 'ImplExternal1' not in class_dict
    assert 'ImplExternal2' not in class_dict
    assert 'ImplExternal3' not in class_dict

    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'


@mock.patch.dict(os.environ,
                 {ENV_VAR: OS_ENV_PATH_SEP.join([EXT_MOD_1, 'asdgasfhsadf',
                                                 'some thing weird', EXT_MOD_2,
                                                 'but still uses sep'])})
def test_external_1_and_2_and_garbage():
    """
    Tests that we can handle invalid module paths intermixed with valid ones
    during search.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 6

    class_dict = {t.__name__: t for t in class_set}

    # Classes we expect to be discovered
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal1' in class_dict
    assert 'ImplExternal2' in class_dict
    assert 'ImplExternal3' in class_dict

    # Check that new classes function as expected
    assert class_dict['ImplFoo']().inst_method('a') == 'fooa'
    assert class_dict['ImplBar']().inst_method('b') == 'barb'
    assert class_dict['ImplDoExport']().inst_method('c') == 'doExportc'
    assert class_dict['ImplExternal1']().inst_method('d') == "external1d"
    assert class_dict['ImplExternal2']().inst_method('e') == "external2e"
    assert class_dict['ImplExternal3']().inst_method('f') == "external3f"


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_LIST})
def test_external_list_helper_value():
    """
    Test that loader handles a list type value assigned to helper attribute.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 5

    class_dict = {t.__name__: t for t in class_set}
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal4' in class_dict
    assert 'ImplExternal5' in class_dict
    assert 'ImplExternal6' not in class_dict


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_INVALID})
def test_external_invalid_helper():
    """
    Test that we error when a helper variable is set to an invalid value.
    """
    with pytest.raises(RuntimeError) as execinfo:
        get_plugins_for_class(DummyInterface)
    assert 'Helper variable set to an invalid value' in str(execinfo.value)


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_ABSTRACT})
def test_external_abstract_impl():
    """
    Test that child classes that are still abstract are not picked up.
    """
    class_set = get_plugins_for_class(DummyInterface)
    assert len(class_set) == 3

    class_dict = {t.__name__: t for t in class_set}
    assert 'ImplFoo' in class_dict
    assert 'ImplBar' in class_dict
    assert 'ImplDoExport' in class_dict
    assert 'ImplExternal7' not in class_dict


class TestExtensionPlugins (object):
    """
    Grouping for unit tests on the entry-point extension based specification
    method.

    Each test can "mock" different extension Entry-points by mocking the
    ``stevedore.ExtensionManager.ENTRY_POINT_CACHE`` dictionary to contain the
    expected ``pkg_resources.Entrypoint`` instances.
    """
    # noinspection PyMethodMayBeStatic
    def setup(self):
        # Remove cache from extensions getter if there is one so we reload from
        # extensions for every test.
        if hasattr(_get_extension_plugin_modules, "ext_manager"):
            delattr(_get_extension_plugin_modules, "ext_manager")

    @mock.patch.dict('smqtk.utils.plugin.ExtensionManager.ENTRY_POINT_CACHE',
                     {EXTENSION_NAMESPACE: [
                         pkg_resources.EntryPoint.parse(
                             'valid-module-with-stuff = '
                             'tests.utils.test_plugin_dir.extension_plugins_1',
                         )
                     ]})
    def test_extension_plugin_valid(self):
        """
        Test that we can load a plugin provided via an entry-point extension.

        This tests as if the following was in the setup.py entry_points dict::

            ...
            'smqtk_plugins':
                'valid-module-with-stuff = '
                'tests.utils.test_plugin_dir.extension_plugins_1'
            ...
        """
        class_set = get_plugins_for_class(DummyInterface)
        class_dict = {t.__name__: t for t in class_set}
        assert len(class_dict) == 4

        assert 'ImplFoo' in class_dict
        assert 'ImplBar' in class_dict
        assert 'ImplDoExport' in class_dict
        assert 'ValidExtensionPlugin' in class_dict
        # Thus unusable class should not be discovered in the same extension
        # module.
        assert 'UnusableExtensionPlugin' not in class_dict

    @mock.patch.dict('smqtk.utils.plugin.ExtensionManager.ENTRY_POINT_CACHE',
                     {EXTENSION_NAMESPACE: [
                         pkg_resources.EntryPoint.parse(
                             'mot-a-module-value = '
                             'tests.utils.test_plugin_dir.extension_plugins_1'
                             ':ValidExtensionPlugin',
                         )
                     ]})
    def test_extension_plugin_import_error(self):
        """
        Test that we are robust to someone providing an extension that cannot be
        loaded due to an exception.

        This tests as if the following was in the setup.py entry_points dict::

            ...
            'smqtk_plugins':
                'mot-a-module-value = '
                'tests.utils.test_plugin_dir.extension_plugins_1'
                ':ValidExtensionPlugin'
            ...
        """
        class_set = get_plugins_for_class(DummyInterface)
        class_dict = {t.__name__: t for t in class_set}
        assert len(class_dict) == 3

        assert 'ImplFoo' in class_dict
        assert 'ImplBar' in class_dict
        assert 'ImplDoExport' in class_dict

        assert 'SomeValidPlugin' not in class_dict

    @mock.patch.dict('smqtk.utils.plugin.ExtensionManager.ENTRY_POINT_CACHE',
                     {EXTENSION_NAMESPACE: [
                         pkg_resources.EntryPoint.parse(
                             "ex1 = "
                             "tests.utils.test_plugin_dir.extension_plugins_1",
                         ),
                         pkg_resources.EntryPoint.parse(
                             "ex2 = "
                             "tests.utils.test_plugin_dir.extension_plugins_2",
                         )
                     ]})
    def test_extension_plugin_multi(self):
        """
        Test that an extension can supply multiple modules, handling errors if
        one provided is bad.

        This tests as if the following was in the setup.py entry_points dict::

            ...
            'smqtk_plugins': [
                "ex1 = tests.utils.test_plugin_dir.extension_plugins_1",
                "ex2 = tests.utils.test_plugin_dir.extension_plugins_2"
            ]
            ...
        """
        class_set = get_plugins_for_class(DummyInterface)
        class_dict = {t.__name__: t for t in class_set}
        assert len(class_dict) == 5

        assert 'ImplFoo' in class_dict
        assert 'ImplBar' in class_dict
        assert 'ImplDoExport' in class_dict
        assert 'ValidExtensionPlugin' in class_dict
        assert 'ValidExtensionPlugin2' in class_dict
        # Thus unusable class should not be discovered in the same extension
        # module.
        assert 'UnusableExtensionPlugin' not in class_dict
