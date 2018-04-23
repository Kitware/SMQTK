from __future__ import division, print_function

import os

import pytest

from smqtk.utils.plugin import (
    get_plugins,
    make_config,
    to_plugin_config,
    from_plugin_config
)
from smqtk.tests.utils.test_configurable_interface import (
    DummyAlgo1, DummyAlgo2
)
from smqtk.tests.utils.test_plugin_dummy_interface import DummyInterface


################################################################################
# Dummy plugin implementation of dummy algorithms with constructor parameters

class DummyPlugin1 (DummyAlgo1, DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return val + '1'


class DummyPlugin2 (DummyAlgo2, DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return val + '2'


###############################################################################
# Fixtures

@pytest.fixture
def get_plugins_dict():
    """ Example return from ``get_plugins``."""
    return get_plugins(__package__, os.path.dirname(__file__), '', '',
                       DummyInterface)


###############################################################################
# Tests

# noinspection PyShadowingNames
def test_make_config(get_plugins_dict):
    """
    Test expected configuration JSON block construction from ``make_config``.
    """
    expected = {
        'type': None,
        'DummyPlugin1': DummyPlugin1.get_default_config(),
        'DummyPlugin2': DummyPlugin2.get_default_config(),
    }
    assert make_config(get_plugins_dict) == expected


def test_to_config():
    i = DummyPlugin1()
    d1 = i.get_config()
    c = to_plugin_config(i)

    expected = {
        'type': 'DummyPlugin1',
        'DummyPlugin1': d1,
    }
    assert c == expected

    # return should update with updates to
    i.foo = 8
    d2 = i.get_config()
    assert d1 != d2
    c = to_plugin_config(i)

    expected = {
        'type': 'DummyPlugin1',
        'DummyPlugin1': d2,
    }
    assert c == expected


# noinspection PyShadowingNames
def test_from_config(get_plugins_dict):
    test_config = {
        'type': 'DummyPlugin1',
        'DummyPlugin1': {'foo': 256, 'bar': 'Some string value'},
        'DummyPlugin2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }

    #: :type: DummyAlgo1
    i = from_plugin_config(test_config, get_plugins_dict)
    assert isinstance(i, DummyPlugin1)
    assert i.foo == 256
    assert i.bar == "Some string value"


# noinspection PyShadowingNames
def test_from_config_missing_type(get_plugins_dict):
    test_config = {
        'DummyAlgo1': {'foo': 256, 'bar': 'Some string value'},
        'DummyAlgo2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_plugin_config(test_config, get_plugins_dict)
    assert "does not have an implementation type specification" \
           in str(execinfo.value)


# noinspection PyShadowingNames
def test_from_config_none_type(get_plugins_dict):
    test_config = {
        'type': None,
        'DummyAlgo1': {'foo': 256, 'bar': 'Some string value'},
        'DummyAlgo2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_plugin_config(test_config, get_plugins_dict)
    assert "No implementation type specified" in str(execinfo.value)


# noinspection PyShadowingNames
def test_from_config_config_label_mismatch(get_plugins_dict):
    test_config = {
        'type': 'not-present-label',
        'DummyAlgo1': {'foo': 256, 'bar': 'Some string value'},
        'DummyAlgo2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_plugin_config(test_config, get_plugins_dict)
    assert "no configuration block was present for that type" \
           in str(execinfo.value)


# noinspection PyShadowingNames
def test_from_config_impl_label_mismatch(get_plugins_dict):
    test_config = {
        'type': 'notAnImpl',
        'DummyAlgo1': {'foo': 256, 'bar': 'Some string value'},
        'DummyAlgo2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_plugin_config(test_config, get_plugins_dict)
    assert "no plugin implementations are available for that type" \
           in str(execinfo.value)
