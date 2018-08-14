from __future__ import division, print_function

import unittest

from smqtk.utils.plugin import (
    make_config,
    to_plugin_config,
    from_plugin_config
)
from smqtk.tests.utils.test_configurable_interface import (
    DummyAlgo1, DummyAlgo2
)


def dummy_getter():
    return {
        'DummyAlgo1': DummyAlgo1,
        'DummyAlgo2': DummyAlgo2,
    }


class TestPluginTools (unittest.TestCase):

    def test_make_config(self):
        self.assertEqual(
            make_config(dummy_getter()),
            {
                'type': None,
                'DummyAlgo1': DummyAlgo1.get_default_config(),
                'DummyAlgo2': DummyAlgo2.get_default_config(),
            }
        )

    def test_to_config(self):
        i = DummyAlgo1()
        d1 = i.get_config()
        c = to_plugin_config(i)
        self.assertEqual(c, {
            'type': 'DummyAlgo1',
            'DummyAlgo1': d1,
        })

        # return should update with updates to
        i.foo = 8
        d2 = i.get_config()
        self.assertNotEqual(d1, d2)
        c = to_plugin_config(i)
        self.assertEqual(c, {
            'type': 'DummyAlgo1',
            'DummyAlgo1': d2,
        })

    def test_from_config(self):
        test_config = {
            'type': 'DummyAlgo1',
            'DummyAlgo1': {'foo': 256, 'bar': 'Some string value'},
            'DummyAlgo2': {
                'child': {'foo': -1, 'bar': 'some other value'},
                'alpha': 1.0,
                'beta': 'euclidean',
            },
            'notAnImpl': {}
        }

        #: :type: DummyAlgo1
        i = from_plugin_config(test_config, dummy_getter())
        self.assertIsInstance(i, DummyAlgo1)
        self.assertEqual(i.foo, 256)
        self.assertEqual(i.bar, 'Some string value')

    def test_from_config_missing_type(self):
        test_config = {
            'DummyAlgo1': {'foo': 256, 'bar': 'Some string value'},
            'DummyAlgo2': {
                'child': {'foo': -1, 'bar': 'some other value'},
                'alpha': 1.0,
                'beta': 'euclidean',
            },
            'notAnImpl': {}
        }
        self.assertRaisesRegexp(
            ValueError,
            "does not have an implementation type specification",
            from_plugin_config,
            test_config, dummy_getter()
        )

    def test_from_config_none_type(self):
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
        self.assertRaisesRegexp(
            ValueError,
            "No implementation type specified",
            from_plugin_config,
            test_config,
            dummy_getter()
        )

    def test_from_config_config_label_mismatch(self):
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
        self.assertRaisesRegexp(
            ValueError,
            "no configuration block was present for that type",
            from_plugin_config,
            test_config,
            dummy_getter()
        )

    def test_from_config_impl_label_mismatch(self):
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
        self.assertRaisesRegexp(
            ValueError,
            "no plugin implementations are available for that type",
            from_plugin_config,
            test_config,
            dummy_getter()
        )
