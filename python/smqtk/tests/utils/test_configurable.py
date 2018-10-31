import pytest

from smqtk.utils.configuration import (
    Configurable,
    make_default_config,
    to_config_dict,
    from_config_dict,
)


################################################################################
# Helper classes for testing

class T1 (Configurable):

    def __init__(self, foo=1, bar='baz'):
        self.foo = foo
        self.bar = bar

    def get_config(self):
        return {
            'foo': self.foo,
            'bar': self.bar,
        }


class T2 (Configurable):
    """
    Semi-standard way to implement nested algorithms.
    Usually, algorithms are associated to a plugin getter method that wraps a
    call to ``smqtk.utils.plugin.get_plugins``, but I'm skipping that for now
    for simplicity.
    """

    @classmethod
    def get_default_config(cls):
        default = super(T2, cls).get_default_config()
        # Replace ``child``, which we know has a default value of type ``T1``
        # with its default config.
        default['child'] = default['child'].get_default_config()
        return default

    @classmethod
    def from_config(cls, config, merge_default=True):
        config['child'] = T1.from_config(config['child'])
        return super(T2, cls).from_config(config)

    def __init__(self, child=T1(), alpha=0.0001, beta='default'):
        # Where child is supposed to be an instance of DummyAlgo1
        self.child = child
        self.alpha = alpha
        self.beta = beta

    def get_config(self):
        return {
            'child': self.child.get_config(),
            'alpha': self.alpha,
            'beta': self.beta
        }


# Set of "available" types for tests below.
T_CLASS_SET = {T1, T2}


################################################################################
# Tests

def test_configurable_default_config():
    """
    Test that constructor arguments are introspected automatically with None
    defaults.
    """
    # Only using class methods, thus noinspection.
    # noinspection PyAbstractClass
    class T (Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, a, b, cat):
            pass
    assert T.get_default_config() == {'a': None, 'b': None, 'cat': None}


def test_configurable_default_config_with_star_args():
    """
    Test that star-stuff shouldn't change anything
    """
    # Only using class methods, thus noinspection.
    # noinspection PyAbstractClass
    class T (Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, a, b, cat, *args, **kwargs):
            pass
    assert T.get_default_config() == {'a': None, 'b': None, 'cat': None}


def test_configurable_default_config_with_default_values():
    """
    Test that default values are correctly introspected.
    """
    # Only using class methods, thus noinspection.
    # noinspection PyAbstractClass
    class T (Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, a, b, c=0, d='foobar'):
            pass
    assert T.get_default_config() == \
           {'a': None, 'b': None, 'c': 0, 'd': 'foobar'}


def test_configurable_default_config_with_default_values_with_star_args():
    """
    Test that star stuff shouldn't change anything when there are default
    values.
    """
    # Only using class methods, thus noinspection.
    # noinspection PyAbstractClass
    class T (Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, a, b, c=0, d='foobar', *args, **kwargs):
            pass
    assert T.get_default_config() == \
           {'a': None, 'b': None, 'c': 0, 'd': 'foobar'}


def test_configurable_classmethod_override_getdefaultconfig():
    """
    Test overriding the class method get_default_config
    """
    assert T1.get_default_config() == {'foo': 1, 'bar': 'baz'}
    assert T2.get_default_config() == {
        'child': {'foo': 1, 'bar': 'baz'},
        'alpha': 0.0001,
        'beta': 'default'
    }


def test_make_default_config():
    """
    Test expected normal operation of ``make_default_config``.
    """
    expected = {
        'type': None,
        'T1': T1.get_default_config(),
        'T2': T2.get_default_config(),
    }
    assert make_default_config(T_CLASS_SET) == expected


def test_to_config_dict():
    """
    Test that ``to_config_dict`` correctly reflects the contents of the config
    return from the instance passed to it.
    """
    i1 = T1()
    c1 = to_config_dict(i1)
    expected1 = {
        'type': 'T1',
        'T1': {
            'foo': 1,
            'bar': 'baz',
        },
    }
    assert c1 == expected1

    # return should update with updates to
    i2 = T1(foo=8)
    c2 = to_config_dict(i2)
    expected2 = {
        'type': 'T1',
        'T1': {
            'foo': 8,
            'bar': 'baz',
        },
    }
    assert c2 == expected2


def test_from_config_dict():
    """
    Test that ``from_config_dict`` correctly creates an instance of the class
    requested by the configuration.
    """
    test_config = {
        'type': 'T1',
        'T1': {'foo': 256, 'bar': 'Some string value'},
        'T2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }

    #: :type: DummyAlgo1
    i = from_config_dict(test_config, T_CLASS_SET)
    assert isinstance(i, T1)
    assert i.foo == 256
    assert i.bar == "Some string value"


def test_from_config_missing_type():
    """
    Test that ``from_config_dict`` throws an exception when no 'type' key is
    present.
    """
    test_config = {
        'T1': {'foo': 256, 'bar': 'Some string value'},
        'T2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_config_dict(test_config, T_CLASS_SET)
    assert "does not have an implementation type specification" \
           in str(execinfo.value)


def test_from_config_none_type():
    """
    Test that appropriate exception is raised when `type` key is None valued.
    """
    test_config = {
        'type': None,
        'T1': {'foo': 256, 'bar': 'Some string value'},
        'T2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_config_dict(test_config, T_CLASS_SET)
    assert "No implementation type specified" in str(execinfo.value)


def test_from_config_config_label_mismatch():
    test_config = {
        'type': 'not-present-label',
        'T1': {'foo': 256, 'bar': 'Some string value'},
        'T2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_config_dict(test_config, T_CLASS_SET)
    assert "no configuration block was present for that type" \
           in str(execinfo.value)


def test_from_config_impl_label_mismatch():
    test_config = {
        'type': 'notAnImpl',
        'T1': {'foo': 256, 'bar': 'Some string value'},
        'T2': {
            'child': {'foo': -1, 'bar': 'some other value'},
            'alpha': 1.0,
            'beta': 'euclidean',
        },
        'notAnImpl': {}
    }
    with pytest.raises(ValueError) as execinfo:
        from_config_dict(test_config, T_CLASS_SET)
    assert "no plugin implementations are available for that type" \
           in str(execinfo.value)
