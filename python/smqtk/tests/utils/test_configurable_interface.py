import unittest

from smqtk.utils.configurable_interface import Configurable


class TestConfigurablePluginInterface (unittest.TestCase):

    def test_default_config(self):
        # Only using class method.
        # noinspection PyAbstractClass
        class T (Configurable):
            # noinspection PyUnusedLocal
            def __init__(self, a, b, cat):
                pass
        self.assertEqual(
            T.get_default_config(),
            {'a': None, 'b': None, 'cat': None}
        )

    def test_default_config_with_star_args(self):
        # Test that star stuff shouldn't change anything

        # Only using class method.
        # noinspection PyAbstractClass
        class T (Configurable):
            # noinspection PyUnusedLocal
            def __init__(self, a, b, cat, *args, **kwargs):
                pass
        self.assertEqual(
            T.get_default_config(),
            {'a': None, 'b': None, 'cat': None}
        )

    def test_default_config_with_default_values(self):
        # Only using class method.
        # noinspection PyAbstractClass
        class T (Configurable):
            # noinspection PyUnusedLocal
            def __init__(self, a, b, c=0, d='foobar'):
                pass
        self.assertEqual(
            T.get_default_config(),
            {'a': None, 'b': None, 'c': 0, 'd': 'foobar'}
        )

    def test_default_config_with_default_values_with_star_args(self):
        # Test that star stuff shouldn't change anything.

        # Only using class method.
        # noinspection PyAbstractClass
        class T (Configurable):
            # noinspection PyUnusedLocal
            def __init__(self, a, b, c=0, d='foobar', *args, **kwargs):
                pass
        self.assertEqual(
            T.get_default_config(),
            {'a': None, 'b': None, 'c': 0, 'd': 'foobar'}
        )


class DummyAlgo1 (Configurable):

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


class DummyAlgo2 (Configurable):
    """
    Semi-standard way to implement nested algorithms.
    Usually, algorithms are associated to a plugin getter method that wraps a
    call to ``smqtk.utils.plugin.get_plugins``, but I'm skipping that for now
    for simplicity.
    """

    @classmethod
    def get_default_config(cls):
        default = super(DummyAlgo2, cls).get_default_config()
        # replace ``child`` with config available config
        default['child'] = default['child'].get_config()
        return default

    @classmethod
    def from_config(cls, config, merge_default=True):
        config['child'] = DummyAlgo1(**config['child'])
        return cls(**config)

    def __init__(self, child=DummyAlgo1(), alpha=0.0001, beta='default'):
        # Where child is supposed to be an instance of DummyAlgo1
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
        self.assertEqual(
            DummyAlgo1.get_default_config(),
            {'foo': 1, 'bar': 'baz'}
        )

    def test_algo2_default(self):
        self.assertEqual(
            DummyAlgo2.get_default_config(),
            {
                'child': {'foo': 1, 'bar': 'baz'},
                'alpha': 0.0001,
                'beta': 'default'
            }
        )
