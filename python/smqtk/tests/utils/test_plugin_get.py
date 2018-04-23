from __future__ import division, print_function

import os

import pytest
from six.moves import mock

from smqtk.utils.plugin import get_plugins, OS_ENV_PATH_SEP
from smqtk.tests.utils.test_plugin_dummy_interface import DummyInterface


###############################################################################
# Test constants

INTERNAL_PLUGIN_DIR = os.path.join(os.path.dirname(__file__),
                                   "test_plugin_dir",
                                   "internal_plugins")

INTERNAL_PLUGIN_MOD_PATH = \
    'smqtk.tests.utils.test_plugin_dir.internal_plugins'

ENV_VAR = "TEST_PLUGIN_MODULE_PATH"
HELP_VAR = "TEST_PLUGIN_CLASS"

EXT_MOD_1 = 'smqtk.tests.utils.test_plugin_dir.external_1'
EXT_MOD_2 = 'smqtk.tests.utils.test_plugin_dir.external_2'
EXT_MOD_LIST = 'smqtk.tests.utils.test_plugin_dir.external_list'
EXT_MOD_INVALID = 'smqtk.tests.utils.test_plugin_dir.external_invalid_helper'
EXT_MOD_ABSTRACT = 'smqtk.tests.utils.test_plugin_dir.external_still_abstract'


def get_plugins_for_class(cls):
    """
    Test standard wrapper on get_plugins call using test constants.
    This is not a fixture due to environment variable mocking.
    """
    return get_plugins(INTERNAL_PLUGIN_MOD_PATH, INTERNAL_PLUGIN_DIR, ENV_VAR,
                       HELP_VAR, cls)


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
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplFoo' in m
    assert 'ImplBar' in m
    assert 'ImplDoExport' in m

    assert m['ImplFoo']().inst_method('a') == 'fooa'
    assert m['ImplBar']().inst_method('b') == 'barb'
    assert m['ImplDoExport']().inst_method('c') == 'doExportc'

    assert 'ImplNotUsable' not in m
    assert 'SomethingElse' not in m
    assert 'ImplNoExport' not in m
    assert 'ImplSkipModule' not in m


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_1})
def test_external_1_only():
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplExternal1' in m
    assert 'ImplExternal2' in m
    assert 'ImplExternal3' not in m

    assert m['ImplExternal1']().inst_method('d') == "external1d"
    assert m['ImplExternal2']().inst_method('e') == "external2e"


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_1+OS_ENV_PATH_SEP})
def test_external_1_with_trailing_sep():
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplExternal1' in m
    assert 'ImplExternal2' in m
    assert 'ImplExternal3' not in m

    assert m['ImplExternal1']().inst_method('d') == "external1d"
    assert m['ImplExternal2']().inst_method('e') == "external2e"


@mock.patch.dict(os.environ, {ENV_VAR: OS_ENV_PATH_SEP+EXT_MOD_1})
def test_external_1_with_leading_sep():
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplExternal1' in m
    assert 'ImplExternal2' in m
    assert 'ImplExternal3' not in m

    assert m['ImplExternal1']().inst_method('d') == "external1d"
    assert m['ImplExternal2']().inst_method('e') == "external2e"


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_2})
def test_external_2_only():
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplExternal1' not in m
    assert 'ImplExternal2' not in m
    assert 'ImplExternal3' in m

    assert m['ImplExternal3']().inst_method('f') == 'external3f'


@mock.patch.dict(os.environ, {ENV_VAR: OS_ENV_PATH_SEP.join([EXT_MOD_1,
                                                             EXT_MOD_2])})
def test_external_1_and_2():
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplExternal1' in m
    assert 'ImplExternal2' in m
    assert 'ImplExternal3' in m

    assert m['ImplExternal1']().inst_method('d') == "external1d"
    assert m['ImplExternal2']().inst_method('e') == "external2e"
    assert m['ImplExternal3']().inst_method('f') == "external3f"


@mock.patch.dict(os.environ, {ENV_VAR: "This is a junk string"})
def test_junk_external_mod():
    # This should skip module it can't find. In this case, the junk string
    # is treated as a python module path, which is invalid. A warning is
    # emitted but the plugin query succeeds as if the invalid chunk didn't
    # exist.
    m = get_plugins_for_class(DummyInterface)

    assert 'ImplFoo' in m
    assert 'ImplBar' in m
    assert 'ImplDoExport' in m

    assert 'ImplExternal1' not in m
    assert 'ImplExternal2' not in m
    assert 'ImplExternal3' not in m


@mock.patch.dict(os.environ, {ENV_VAR: OS_ENV_PATH_SEP.join([EXT_MOD_1,
                                                             EXT_MOD_2,
                                                             'asdgasfhsadf',
                                                             'some thing weird',
                                                             'but still uses sep'])})
def test_external_1_and_2_and_garbage():
    """
    Tests that we can handle invalid module paths during search.
    """
    m = get_plugins_for_class(DummyInterface)

    assert len(m) == 6
    assert 'ImplFoo' in m
    assert 'ImplBar' in m
    assert 'ImplDoExport' in m
    assert 'ImplExternal1' in m
    assert 'ImplExternal2' in m
    assert 'ImplExternal3' in m

    assert m['ImplFoo']().inst_method('a') == 'fooa'
    assert m['ImplBar']().inst_method('b') == 'barb'
    assert m['ImplDoExport']().inst_method('c') == 'doExportc'
    assert m['ImplExternal1']().inst_method('d') == "external1d"
    assert m['ImplExternal2']().inst_method('e') == "external2e"
    assert m['ImplExternal3']().inst_method('f') == "external3f"


@mock.patch.dict(os.environ, {ENV_VAR: EXT_MOD_LIST})
def test_external_list_helper_value():
    """
    Test that loader handles a list type value.
    """
    m = get_plugins_for_class(DummyInterface)
    assert len(m) == 5
    assert 'ImplFoo' in m
    assert 'ImplBar' in m
    assert 'ImplDoExport' in m
    assert 'ImplExternal4' in m
    assert 'ImplExternal5' in m


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
    m = get_plugins_for_class(DummyInterface)
    assert len(m) == 3
    assert 'ImplFoo' in m
    assert 'ImplBar' in m
    assert 'ImplDoExport' in m
