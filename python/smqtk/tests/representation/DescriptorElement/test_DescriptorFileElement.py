import mock
import nose.tools as ntools
import unittest

import numpy

from smqtk.representation.descriptor_element.local_elements import DescriptorFileElement


__author__ = "paul.tunison@kitware.com"


class TestDescriptorFileElement (unittest.TestCase):

    def test_configuration1(self):
        default_config = DescriptorFileElement.get_default_config()
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

    def test_vec_filepath_generation(self):
        d = DescriptorFileElement('test', 'abcd', '/base', 4)
        ntools.assert_equal(d._vec_filepath,
                            '/base/a/b/c/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base', 2)
        ntools.assert_equal(d._vec_filepath,
                            '/base/ab/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base', 1)
        ntools.assert_equal(d._vec_filepath,
                            '/base/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base', 0)
        ntools.assert_equal(d._vec_filepath,
                            '/base/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base')
        ntools.assert_equal(d._vec_filepath,
                            '/base/test.abcd.vector.npy')

    @mock.patch('smqtk.representation.descriptor_element.local_elements'
                '.numpy.save')
    @mock.patch('smqtk.representation.descriptor_element.local_elements'
                '.file_utils.safe_create_dir')
    def test_vector_set(self, mock_scd, mock_save):
        d = DescriptorFileElement('test', 1234, '/base', 4)
        ntools.assert_equal(d._vec_filepath,
                            '/base/1/2/3/test.1234.vector.npy')

        v = numpy.zeros(16)
        d.set_vector(v)
        mock_scd.assert_called_with('/base/1/2/3')
        mock_save.assert_called_with('/base/1/2/3/test.1234.vector.npy', v)

    @mock.patch('smqtk.representation.descriptor_element.local_elements'
                '.numpy.load')
    def test_vector_get(self, mock_load):
        d = DescriptorFileElement('test', 1234, '/base', 4)
        ntools.assert_false(d.has_vector())
        ntools.assert_is(d.vector(), None)

        d.has_vector = mock.Mock(return_value=True)
        ntools.assert_true(d.has_vector())
        v = numpy.zeros(16)
        mock_load.return_value = v
        numpy.testing.assert_equal(d.vector(), v)
