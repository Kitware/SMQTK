import unittest

import mock
import nose.tools as ntools

from smqtk.algorithms.descriptor_generator.colordescriptor.colordescriptor import (
    ColorDescriptor_Image_csift  # arbitrary leaf class
)

__author__ = "paul.tunison@kitware.com"


class TestColorDescriptor (unittest.TestCase):

    @mock.patch('smqtk.algorithms.descriptor_generator.colordescriptor.colordescriptor'
                '.safe_create_dir')
    def test_configuration(self, mock_scd):
        default_config = ColorDescriptor_Image_csift.get_default_config()
        default_config['model_directory'] = '/some/path/models/'
        default_config['work_directory'] = '/some/path/work/'

        inst = ColorDescriptor_Image_csift.from_config(default_config)
        ntools.assert_equal(
            default_config,
            inst.get_config()
        )
        inst2 = ColorDescriptor_Image_csift.from_config(inst.get_config())
        ntools.assert_equal(inst.get_config(), inst2.get_config())
        ntools.assert_equal(inst._model_dir, inst2._model_dir)
        ntools.assert_equal(inst._work_dir, inst2._work_dir)
        ntools.assert_equal(inst._kmeans_k, inst2._kmeans_k)
        ntools.assert_equal(inst._flann_target_precision,
                            inst2._flann_target_precision)
        ntools.assert_equal(inst._flann_sample_fraction,
                            inst2._flann_sample_fraction)
        ntools.assert_equal(inst._flann_autotune, inst2._flann_autotune)
        ntools.assert_equal(inst._use_sp, inst2._use_sp)
        ntools.assert_equal(inst._rand_seed, inst2._rand_seed)
