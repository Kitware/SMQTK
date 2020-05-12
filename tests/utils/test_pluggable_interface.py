import os

import pytest
import unittest.mock as mock

from smqtk.utils.plugin import Pluggable, NotUsableError


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


class DummyImpl (Pluggable):

    TEST_USABLE = True

    @classmethod
    def is_usable(cls):
        return cls.TEST_USABLE


class DummyImplSub (DummyImpl):
    pass


###############################################################################
# Tests

@mock.patch.object(DummyImpl, 'TEST_USABLE', new_callable=mock.PropertyMock)
def test_construct_when_usable(m_TEST_USABLE):
    # Construction should happen without incident
    m_TEST_USABLE.return_value = True
    DummyImpl()


@mock.patch.object(DummyImpl, 'TEST_USABLE', new_callable=mock.PropertyMock)
def test_construct_when_not_usable(m_TEST_USABLE):
    # Should raise NotUsableError exception on construction.
    m_TEST_USABLE.return_value = False
    with pytest.raises(NotUsableError):
        DummyImpl()


def test_get_impls_expected_defaults():
    """
    Test that the correct package and containing module directory is correct
    for the dummy plugin.
    """
    mock_return_value = 'mock return'
    with mock.patch('smqtk.utils.plugin.get_plugins') as m_get_plugins:
        m_get_plugins.return_value = mock_return_value
        assert DummyImpl.get_impls() == mock_return_value
        m_get_plugins.assert_called_once_with(DummyImpl,
                                              'SMQTK_PLUGIN_PATH',
                                              'SMQTK_PLUGIN_CLASS',
                                              # Default ``warn`` value
                                              warn=True,
                                              # Default ``reload_modules`` value
                                              reload_modules=False)


def test_get_impls_do_reload():
    """
    Test passing change to ``reload_modules`` argument.
    """
    mock_return_value = 'mock return'
    with mock.patch('smqtk.utils.plugin.get_plugins') as m_get_plugins:
        m_get_plugins.return_value = mock_return_value
        assert DummyImpl.get_impls(reload_modules=True) == mock_return_value
        m_get_plugins.assert_called_once_with(DummyImpl,
                                              'SMQTK_PLUGIN_PATH',
                                              'SMQTK_PLUGIN_CLASS',
                                              warn=True,
                                              reload_modules=True)


@mock.patch.object(DummyImpl, 'PLUGIN_HELPER_VAR',
                   new_callable=mock.PropertyMock)
@mock.patch.object(DummyImpl, 'PLUGIN_ENV_VAR',
                   new_callable=mock.PropertyMock)
def test_get_impls_change_vars(m_env_var_prop, m_helper_var_prop):
    """
    Test that changes to env/helper vars propagates to call to underlying
    ``get_plugins`` functional call.
    """
    expected_return_value = 'mock return'
    expected_env_var = m_env_var_prop.return_value = "new test env var"
    expected_helper_var = m_helper_var_prop.return_value = "new test helper var"
    with mock.patch('smqtk.utils.plugin.get_plugins') as m_get_plugins:
        m_get_plugins.return_value = expected_return_value
        assert DummyImpl.get_impls() == expected_return_value
        m_get_plugins.assert_called_once_with(DummyImpl,
                                              expected_env_var,
                                              expected_helper_var,
                                              warn=True,
                                              reload_modules=False)


# Make sure there is not something in the environment to mess with this test.
@mock.patch.dict(os.environ, {DummyImpl.PLUGIN_ENV_VAR: ""})
def test_get_impls_implemented_classes():
    """
    DESIGN TEST: Test that a leaf class (i.e. full implementation of an abstract
    base class) still returns something when the ``get_impls`` class method is
    called, just that its empty.  If there are sub-classes to a fully
    implemented class, then its discovered sub-classes should be returned.
    """
    expected = {DummyImplSub}
    assert DummyImpl.get_impls() == expected

    expected = set()
    assert DummyImplSub.get_impls() == expected
