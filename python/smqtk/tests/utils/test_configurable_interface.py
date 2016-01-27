
import nose.tools as ntools
import unittest

from smqtk.utils.configurable_interface import Configurable


__author__ = "paul.tunison@kitware.com"


class TestConfigurablePluginInterface (unittest.TestCase):

    def test_default_config(self):
        class T1 (Configurable):
            def __init__(self, a, b, cat):
                pass
        ntools.assert_equal(
            T1.get_default_config(),
            {'a': None, 'b': None, 'cat': None}
        )

        # star stuff shouldn't change anything
        class T2 (Configurable):
            def __init__(self, a, b, cat, *args, **kwargs):
                pass
        ntools.assert_equal(
            T2.get_default_config(),
            {'a': None, 'b': None, 'cat': None}
        )

    def test_default_config_with_default_values(self):
        class T1 (Configurable):
            def __init__(self, a, b, c=0, d='foobar'):
                pass
        ntools.assert_equal(
            T1.get_default_config(),
            {'a': None, 'b': None, 'c': 0, 'd': 'foobar'}
        )

        # star stuff shouldn't change anything
        class T2 (Configurable):
            def __init__(self, a, b, c=0, d='foobar', *args, **kwargs):
                pass
        ntools.assert_equal(
            T2.get_default_config(),
            {'a': None, 'b': None, 'c': 0, 'd': 'foobar'}
        )


class TestAlgo1 (Configurable):

    def __init__(self, foo=1, bar='baz'):
        self.foo = foo
        self.bar = bar

    def __eq__(self, other):
        return self.foo == other.foo and self.bar == other.bar

    def __ne__(self, other):
        return not (self == other)

    def get_config(self):
        return {
            'foo': self.foo,
            'bar': self.bar,
        }


class TestAlgo2 (Configurable):
    """
    Semi-standard way to implement nested algorithms.
    Usually, algorithms are associated to a plugin getter method that wraps a
    call to ``smqtk.utils.plugin.get_plugins``, but I'm skipping that for now
    for simplicity.
    """

    @classmethod
    def get_default_config(cls):
        default = super(TestAlgo2, cls).get_default_config()
        # replace ``child`` with config available config
        default['child'] = default['child'].get_config()
        return default

    @classmethod
    def from_config(cls, config, merge_default=True):
        config['child'] = TestAlgo1(**config['child'])
        return cls(**config)

    def __init__(self, child=TestAlgo1(), alpha=0.0001, beta='default'):
        # Where child is supposed to be an instance of TestAlgo1
        self.child = child
        self.alpha = alpha
        self.beta = beta

    def __eq__(self, other):
        return (
            self.child == other.child and
            self.alpha == other.alpha and
            self.beta == other.beta
        )

    def get_config(self):
        return {
            'child': self.child.get_config(),
            'alpha': self.alpha,
            'beta': self.beta
        }


class TestConfigurablePluginTestImpl (unittest.TestCase):

    def test_algo1_default(self):
        ntools.assert_equal(
            TestAlgo1.get_default_config(),
            {'foo': 1, 'bar': 'baz'}
        )

    def test_algo2_default(self):
        ntools.assert_equal(
            TestAlgo2.get_default_config(),
            {
                'child': {'foo': 1, 'bar': 'baz'},
                'alpha': 0.0001,
                'beta': 'default'
            }
        )
