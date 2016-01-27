import nose.tools as ntools
import unittest

from smqtk.representation.data_set.file_set import DataFileSet


__author__ = "paul.tunison@kitware.com"


class TestDataFileSet (unittest.TestCase):

    def test_configuration(self):
        default_config = DataFileSet.get_default_config()
        ntools.assert_equal(default_config, {
            'root_directory': None,
            'uuid_chunk': 10,
            'pickle_protocol': -1,
        })

        default_config['root_directory'] = '/some/dir'
        inst1 = DataFileSet.from_config(default_config)
        ntools.assert_equal(default_config, inst1.get_config())

        inst2 = DataFileSet.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)
