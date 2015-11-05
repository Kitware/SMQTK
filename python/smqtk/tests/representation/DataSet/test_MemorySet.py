import nose.tools as ntools
import unittest

from smqtk.representation.data_set.memory_set import DataMemorySet


__author__ = "paul.tunison@kitware.com"


class TestDataFileSet (unittest.TestCase):

    def test_configuration(self):
        default_config = DataMemorySet.get_default_config()
        expected_config = {"file_cache": None}
        ntools.assert_equal(default_config, expected_config)

        inst1 = DataMemorySet.from_config(default_config)
        # idempotency
        ntools.assert_equal(default_config, inst1.get_config())

        inst2 = DataMemorySet.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)
