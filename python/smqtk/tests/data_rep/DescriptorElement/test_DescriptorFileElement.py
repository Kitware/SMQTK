import nose.tools as ntools
import unittest

from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorFileElement


__author__ = 'purg'


class TestDescriptorFileElement (unittest.TestCase):

    def test_configuration(self):
        default_config = DescriptorFileElement.default_config()
        ntools.assert_equal(default_config,
                            {
                                'save_dir': None,
                                'subdir_split': None,
                            })

        default_config['save_dir'] = '/some/path/somewhere'
        default_config['subdir_split'] = 4

        inst1 = DescriptorFileElement.from_config(default_config,
                                                  'test', 'abcd')
        ntools.assert_equal(default_config, inst1.get_config())
        ntools.assert_equal(inst1._save_dir, '/some/path/somewhere')
        ntools.assert_equal(inst1._subdir_split, 4)

        # vector-based equality
        inst2 = DescriptorFileElement.from_config(inst1.get_config(),
                                                  'test', 'abcd')
        ntools.assert_equal(inst1, inst2)
