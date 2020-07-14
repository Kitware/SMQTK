from __future__ import division, print_function
import inspect
import os
import pickle
import unittest

import PIL.Image
import mock
import numpy
from matplotlib.cbook import get_sample_data

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.algorithms.descriptor_generator.pytorchdescriptor.pytorch_model_descriptors.py import \
    torch, PytorchModelDescriptor
# Testing protected helper function
# noinspection PyProtectedMember
from smqtk.algorithms.descriptor_generator.caffe_descriptor import \
    _process_load_img_array
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.utils.configuration import to_config_dict

from tests import TEST_DATA_DIR


if PytorchModelDescriptor.is_usable():

    class TestPytorchModelDescriptor (unittest.TestCase):

        hopper_image_fp = get_sample_data('grace_hopper.png', asfileobj=False)
        hopper_torch_res18_avgpool_descr_fp = os.path.join(
            TEST_DATA_DIR, 'Hopper.resnet18_avgpool_output.npy'
        )

        self.model_name_elem = 'resnet18'
        self.return_layer_elem = 'avgpool'
        self.norm_mean_elem = [0.485, 0.456, 0.406]
        self.norm_std_elem = [0.229, 0.224, 0.225]
        self.pretrained = True

        # Dummy pytorch configuration files + weights
        dummy_model_name = DataFileElement('dummy_model')
        dummy_return_layer = DataFileElement('layer.position')
        dummy_mean = [0.01, -0.4, 0.20]
        dummy_std = [0, -0.2, -0.6]
        dummy_model_weights = DataFileElement(
            os.path.join(TEST_DATA_DIR, 'test_w_res18.pth'),
            readonly=True
        )

        @classmethod
        def setup_class(cls):
            cls.model_name = 'resnet18'
            cls.return_layer = 'avgpool'
            cls.norm_mean = [0.485, 0.456, 0.406] 
            cls.norm_std = [0.229, 0.224, 0.225]
            if not torch.cuda.is_available():
                cls.use_gpu = False

        def test_impl_findable(self):
            self.assertIn(PytorchModelDescriptor,
                          DescriptorGenerator.get_impls())

        @mock.patch('smqtk.algorithms.descriptor_generator.'
                    '.PytorchModelDescriptor')
        def test_get_config(self, _m_cdg_setupNetwork):
            # Mocking set_network
            expected_params = {
                'model_name': 'dummy-network',
                'return_layer': 'layer name',
                'custom_model_arch': True,
                'weights_filepath': None,
                'norm_mean': [0, 0, -0.5],
                'norm_std': [0.2, 0.3, 1],
                'use_gpu': False,
                'batch_size': 777,
                'pretrained': False,
            }
            # make sure that we're considering all constructor parameter options
            expected_param_keys = \
                set(inspect.getargspec(PytorchModelDescriptor.__init__)
                           .args[1:])
            self.assertSetEqual(set(expected_params.keys()),
                                expected_param_keys)
            g = PytorchModelDescriptor(**expected_params)
            self.assertEqual(g.get_config(), expected_params)

        @mock.patch('smqtk.algorithms.descriptor_generator.'
                    '.PytorchModelDescriptor')
        def test_no_internal_compute_descriptor(self, _m_cdg_setupNetwork):
            # This implementation's descriptor computation logic sits in async
            # method override due to Pytorch's natural multi-element computation
            # interface. Thus, ``_compute_descriptor`` should not be
            # implemented.

            # Passing purposefully bag constructor parameters and ignoring
            # noinspection PyTypeChecker
            g = PytorchModelDescriptor()
            self.assertRaises(
                NotImplementedError,
                g._compute_descriptor, None
            ) 

        def test_compute_descriptor_dummy_model(self):
            # Pytorch dummy network interaction test Lenna image)

            # Construct network with an empty model just to see that our
            # interaction with the Pytorch API is successful. We expect a
            # zero-valued descriptor vector.
            self.assertRaises(
                AssertionError,
                PytorchModelDescriptor(model_name=self.dummy_model_name), 
            )

        @unittest.skipUnless(DataUrlElement.is_usable(),
                             "URL resolution not functional")
        def test_compute_descriptor_from_url_hopper_description(self):
            # Caffe AlexNet interaction test (Grace Hopper image)
            # This is a long test since it has to download data for remote URIs
            d = PytorchModelDescriptor(
                    self.model_name_elem,
                    self.return_layer_elem,
                    None, None,
                    self.norm_mean_elem,
                    self.norm_std_elem,
                    True, 32, self.pretrained)      
            hopper_elem = DataFileElement(self.hopper_image_fp, readonly=True)
            expected_descr = numpy.load(self.hopper_torch_res18_avgpool_descr_fp)
            descr = d.compute_descriptor(hopper_elem).vector()
            numpy.testing.assert_allclose(descr, expected_descr, atol=1e-4)

        def test_compute_descriptor_async_no_data(self):
            # Should get a ValueError when given no descriptors to async method
            g = PytorchModelDescriptor(
                    self.model_name_elem,
                    self.return_layer_elem,
                    None, None,
                    self.norm_mean_elem, 
                    self.norm_std_elem,
                    True, 32, self.pretrained)
            self.assertRaises(
                ValueError,
                g.compute_descriptor_async, []
            )

       def test_weights_loaded_to_model(self):
           # Should get a ValueError when the network weights weights are not 
           # loaded to the network.
           self.assertRaises(
                ValueError,
                PytorchModelDescriptor(
                    self.model_name_elem,
                    self.return_layer_elem,
                    None, None,
                    self.norm_mean_elem,
                    self.norm_std_elem,
                    True, 32, False))

       def test_return_layer_from_network(self):
           # Should get a KeyError when  the network does not contain
           # the given return layer 
           self.assertRaises(
                KeyError,
                PytorchModelDescriptor(
                    self.model_name_elem,
                    self.dummy_return_layer,
                    None, None,
                    self.norm_mean_elem,
                    self.norm_std_elem,
                    True, 32, True))

