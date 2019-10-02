from __future__ import division, print_function
import unittest

import mock
import pytest

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.algorithms.descriptor_generator.colordescriptor.colordescriptor \
    import ColorDescriptor_Image_csift  # arbitrary leaf class
from smqtk.utils.configuration import configuration_test_helper


@pytest.mark.skipif(not ColorDescriptor_Image_csift.is_usable(),
                    reason="ColorDescriptor generator is not currently usable")
class TestColorDescriptor (unittest.TestCase):

    def test_impl_findable(self):
        self.assertIn(ColorDescriptor_Image_csift.__name__,
                      DescriptorGenerator.get_impls())

    @mock.patch('smqtk.algorithms.descriptor_generator'
                '.colordescriptor.colordescriptor.safe_create_dir')
    def test_configuration(self, _mock_scd):
        i = ColorDescriptor_Image_csift(
            model_directory='test model dir',
            work_directory='test work dir',
            model_gen_descriptor_limit=123764,
            kmeans_k=42, flann_distance_metric='hik',
            flann_target_precision=0.92, flann_sample_fraction=0.71,
            flann_autotune=True, random_seed=7, use_spatial_pyramid=True,
            parallel=3,
        )
        for inst in configuration_test_helper(i):
            assert inst._model_dir == 'test model dir'
            assert inst._work_dir == 'test work dir'
            assert inst._model_gen_descriptor_limit == 123764
            assert inst._kmeans_k == 42
            assert inst._flann_distance_metric == 'hik'
            assert inst._flann_target_precision == 0.92
            assert inst._flann_sample_fraction == 0.71
            assert inst._flann_autotune == True
            assert inst._rand_seed == 7
            assert inst._use_sp == True
            assert inst.parallel == 3
