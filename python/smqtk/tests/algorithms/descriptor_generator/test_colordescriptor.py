from __future__ import division, print_function
import unittest

import mock

from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.algorithms.descriptor_generator.colordescriptor.colordescriptor \
    import ColorDescriptor_Image_csift  # arbitrary leaf class


if ColorDescriptor_Image_csift.is_usable():

    class TestColorDescriptor (unittest.TestCase):

        def test_impl_findable(self):
            self.assertIn(ColorDescriptor_Image_csift.__name__,
                          get_descriptor_generator_impls())

        @mock.patch('smqtk.algorithms.descriptor_generator'
                    '.colordescriptor.colordescriptor'
                    '.file_utils.safe_create_dir')
        def test_configuration(self, _mock_scd):
            default_config = ColorDescriptor_Image_csift.get_default_config()
            default_config['model_directory'] = '/some/path/models/'
            default_config['work_directory'] = '/some/path/work/'

            inst = ColorDescriptor_Image_csift.from_config(default_config)
            self.assertEqual(
                default_config,
                inst.get_config()
            )
            inst2 = ColorDescriptor_Image_csift.from_config(inst.get_config())
            self.assertEqual(inst.get_config(), inst2.get_config())
            self.assertEqual(inst._model_dir, inst2._model_dir)
            self.assertEqual(inst._work_dir, inst2._work_dir)
            self.assertEqual(inst._kmeans_k, inst2._kmeans_k)
            self.assertEqual(inst._flann_target_precision,
                             inst2._flann_target_precision)
            self.assertEqual(inst._flann_sample_fraction,
                             inst2._flann_sample_fraction)
            self.assertEqual(inst._flann_autotune, inst2._flann_autotune)
            self.assertEqual(inst._use_sp, inst2._use_sp)
            self.assertEqual(inst._rand_seed, inst2._rand_seed)
