
import nose.tools as ntools
import unittest

from smqtk.representation.data_element_impl.memory_element import DataMemoryElement


__author__ = 'purg'


class TestDataMemoryElement (unittest.TestCase):

    def test_configuration(self):
        default_config = DataMemoryElement.get_default_config()
        ntools.assert_equal(default_config,
                            {'bytes': None, 'content_type': None})

        default_config['bytes'] = 'Hello World.'
        default_config['content_type'] = 'text/plain'
        inst1 = DataMemoryElement.from_config(default_config)
        ntools.assert_equal(default_config, inst1.get_config())
        ntools.assert_equal(inst1._bytes, 'Hello World.')
        ntools.assert_equal(inst1._content_type, 'text/plain')

        inst2 = DataMemoryElement.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)
