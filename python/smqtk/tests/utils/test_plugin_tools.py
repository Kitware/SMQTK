
import nose.tools as ntools
import unittest

from smqtk.utils.plugin import *
from smqtk.tests.utils.test_configurable_interface import (
    TestAlgo1, TestAlgo2
)


__author__ = "paul.tunison@kitware.com"


def dummy_getter():
    return {
        'TestAlgo1': TestAlgo1,
        'TestAlgo2': TestAlgo2,
    }


class TestPluginTools (unittest.TestCase):

    def test_make_config(self):
        ntools.assert_equal(
            make_config(dummy_getter()),
            {
                'type': None,
                'TestAlgo1': TestAlgo1.get_default_config(),
                'TestAlgo2': TestAlgo2.get_default_config(),
            }
        )

    def test_to_config(self):
        i = TestAlgo1()
        d1 = i.get_config()
        c = to_plugin_config(i)
        ntools.assert_equal(c, {
            'type': 'TestAlgo1',
            'TestAlgo1': d1,
        })

        # return should update with updates to
        i.foo = 8
        d2 = i.get_config()
        ntools.assert_not_equal(d1, d2)
        c = to_plugin_config(i)
        ntools.assert_equal(c, {
            'type': 'TestAlgo1',
            'TestAlgo1': d2,
        })

    def test_from_config(self):
        test_config = {
            'type': 'TestAlgo1',
            'TestAlgo1': {'foo': 256, 'bar': 'Some string value'},
            'TestAlgo2': {
                'child': {'foo': -1, 'bar': 'some other value'},
                'alpha': 1.0,
                'beta': 'euclidean',
            },
            'notAnImpl': {}
        }

        #: :type: TestAlgo1
        i = from_plugin_config(test_config, dummy_getter())
        ntools.assert_is_instance(i, TestAlgo1)
        ntools.assert_equal(i.foo, 256)
        ntools.assert_equal(i.bar, 'Some string value')
