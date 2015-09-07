import nose.tools as ntools
import unittest

from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorMemoryElement


__author__ = 'purg'


class TestDescriptorMemoryElement (unittest.TestCase):

    def test_configuration(self):
        default_config = DescriptorMemoryElement.default_config()
        ntools.assert_equal(default_config, {})

        inst1 = DescriptorMemoryElement.from_config(default_config, 'test', 'a')
        ntools.assert_equal(default_config, inst1.get_config())
        ntools.assert_equal(inst1.type(), 'test')
        ntools.assert_equal(inst1.uuid(), 'a')

        # vector-based equality
        inst2 = DescriptorMemoryElement.from_config(inst1.get_config(),
                                                    'test', 'a')
        ntools.assert_equal(inst1, inst2)
