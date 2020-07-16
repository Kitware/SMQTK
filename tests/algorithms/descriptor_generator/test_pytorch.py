from __future__ import division, print_function
import inspect
import os
import unittest

import six
import PIL.Image
import numpy

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.algorithms.descriptor_generator.pytorchdescriptor.pytorch_model_descriptors import \
    torch, PytorchModelDescriptor
from smqtk.representation.data_element.memory_element import DataMemoryElement

from tests import TEST_DATA_DIR
import pytest

if PytorchModelDescriptor.is_usable():

    class TestPytorchModelDescriptor (unittest.TestCase):

        lenna_image_fp = os.path.join(TEST_DATA_DIR,'Lenna.png')
        lenna_torch_res18_avgpool_descr_fp = os.path.join(
            TEST_DATA_DIR, 'Lenna.resnet18_avgpool_output.npy'
        )

        model_name_elem = 'resnet18'
        return_layer_elem = 'avgpool'
        norm_mean_elem = [0.485, 0.456, 0.406]
        norm_std_elem = [0.229, 0.224, 0.225]
        pretrained = True
        resnet18_avgpool_weights = os.path.join(
            TEST_DATA_DIR,'resnet18_avgpool_weights_torch.pth')       
 
        # Dummy pytorch configuration files + weights
        dummy_model_name = 'dummy_model'
        dummy_return_layer = 'junk_layer'

        @classmethod
        def setup_class(cls):
            cls.model_name = 'resnet18'
            cls.return_layer = 'avgpool'
            cls.input_dim = (224, 224)
            cls.norm_mean = [0.485, 0.456, 0.406] 
            cls.norm_std = [0.229, 0.224, 0.225]
            if not torch.cuda.is_available():
                cls.use_gpu = False

        def test_impl_findable(self):
            self.assertIn(PytorchModelDescriptor,
                          DescriptorGenerator.get_impls())

        def test_get_config(self):
            # Mocking set_network
            expected_params = {
                'model_name': 'resnet18',
                'return_layer': 'avgpool',
                'custom_model_arch': False,
                'weights_filepath': None,
                'input_dim': (24, 996),
                'norm_mean': [0, 0, -0.5],
                'norm_std': [0.2, 0.3, 1],
                'use_gpu': True,
                'batch_size': 777,
                'pretrained': True,
            }
            g = PytorchModelDescriptor(**expected_params)
            self.assertEqual(g.get_config(), expected_params)

        def test_no_internal_compute_descriptor(self):
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

            # Construct network with an dummy model. 
            # We expect an AsserterionError
            self.assertRaises(
                AssertionError,
                PytorchModelDescriptor, model_name = self.dummy_model_name)

        @unittest.skipUnless(DataMemoryElement.is_usable(),
                             "Memory element not functional")
        def test_compute_descriptor_lenna_description(self):
            # Pytorch ResNet interaction test (Lenna image)
            # This is a long test since it has to compute descriptors.
            expected_descr = numpy.load(self.lenna_torch_res18_avgpool_descr_fp)
            d = PytorchModelDescriptor(
                    self.model_name_elem,
                    self.return_layer_elem,
                    None, None, self.input_dim,
                    self.norm_mean_elem,
                    self.norm_std_elem,
                    True, 1, self.pretrained)     
            im = PIL.Image.open(self.lenna_image_fp)
            buff = six.BytesIO()
            (im).save(buff, format="bmp")
            de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/bmp') 
            descr = (d.compute_descriptor(de)).vector()
            numpy.testing.assert_allclose(expected_descr, descr, atol=1e-4)
         
        @unittest.skipUnless(DataMemoryElement.is_usable(),
                             "Memory element not functional")
        def test_load_image_data(self):
            # Testing if image can be loaded and throw an error if uuid is 
            # not automatically generated.
            buff = six.BytesIO()
            im = PIL.Image.open(self.lenna_image_fp)
            (im).save(buff, format="bmp")
            de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/bmp')
            with pytest.raises(AssertionError):
                assert not (de.uuid())  

        def test_compute_descriptor_async_no_data(self):
            # Should get a ValueError when given no descriptors to async method
            g = PytorchModelDescriptor(
                    self.model_name_elem,
                    self.return_layer_elem,
                    None, None, self.input_dim,
                    self.norm_mean_elem, 
                    self.norm_std_elem,
                    True, 32, self.pretrained)
            self.assertRaises(
                ValueError,
                g.compute_descriptor_async, []
            )

        def test_loading_custom_weights_model(self):
            # Should get a ValueError when the network weights are not 
            # loaded to the network or junk weights loaded.
            with pytest.raises(ValueError):
                g = PytorchModelDescriptor(custom_model_arch=None, \
                             weights_filepath=None, pretrained=False)

        def test_weights_loaded_to_model(self):
            # Should fail when the network weights with pretrained flag 
            # loaded are not the imagenet pretrained weights.
            d = PytorchModelDescriptor(
                    self.model_name_elem,
                    self.return_layer_elem,
                    None, None, self.input_dim,
                    self.norm_mean_elem,
                    self.norm_std_elem,
                    True, 1, self.pretrained)
            imagenet_weights = torch.load(self.resnet18_avgpool_weights)  
            d.model.state_dict() == pytest.approx(imagenet_weights, rel=1e-6, abs=1e-12)

        def test_return_layer_from_network(self):
            # Should get a KeyError when  the network does not contain
            # the given return layer 
            with pytest.raises(KeyError):
                g = PytorchModelDescriptor(
                    self.model_name_elem,
                    self.dummy_return_layer,
                    None, None, self.input_dim,
                    self.norm_mean_elem,
                    self.norm_std_elem,
                    True, 32, True)

