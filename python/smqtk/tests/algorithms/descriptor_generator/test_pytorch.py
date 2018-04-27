import inspect
import os
import pickle
import unittest

import mock
import numpy

from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.algorithms.descriptor_generator.pytorch_descriptor import \
     PytorchDescriptorGenerator
from smqtk.representation.data_element import from_uri
from smqtk.tests import TEST_DATA_DIR

from torchvision import models, transforms

if PytorchDescriptorGenerator.is_usable():

    class TestPytorchDesctriptorGenerator (unittest.TestCase):

        lenna_image_fp = os.path.join(TEST_DATA_DIR, 'Lenna.png')
        def setUp(self):
            self.model_cls_name = 'ImageNet_ResNet50'
            self.use_GPU = False
            self.expected_params = {
                'model_cls_name': self.model_cls_name,
                'model_uri': None,
                'resize_val': 256,
                'batch_size': 8,
                'use_gpu': self.use_GPU,
                'in_gpu_device_id': None,
            }

        def test_impl_findable(self):
            self.assertIn(PytorchDescriptorGenerator.__name__,
                                 get_descriptor_generator_impls())


        @mock.patch('smqtk.algorithms.descriptor_generator.pytorch_descriptor'
                    '.PytorchDescriptorGenerator._setup_network')
        def test_get_config(self, m_cdg_setupNetwork):
            # Mocking set_network so we don't have to worry about actually
            # initializing any pytorch things for this test.

            # make sure that we're considering all constructor parameter options
            expected_param_keys = \
                set(inspect.getfullargspec(PytorchDescriptorGenerator.__init__)
                           .args[1:])
            self.assertSetEqual(set(self.expected_params.keys()),
                                        expected_param_keys)
            g = PytorchDescriptorGenerator(**self.expected_params)
            self.assertEqual(g.get_config(), self.expected_params)


        @mock.patch('smqtk.algorithms.descriptor_generator.pytorch_descriptor'
                    '.PytorchDescriptorGenerator._setup_network')
        def test_pickle_save_restore(self, m_cdg_setupNetwork):

            g = PytorchDescriptorGenerator(**self.expected_params)
            # Initialization sets up the network on construction.
            self.assertEqual(m_cdg_setupNetwork.call_count, 1)

            g_pickled = pickle.dumps(g, -1)
            g2 = pickle.loads(g_pickled)
            # Network should be setup for second class class just like in
            # initial construction.
            self.assertEqual(m_cdg_setupNetwork.call_count, 2)
            self.assertIsInstance(g2, PytorchDescriptorGenerator)


        def test_copied_descriptorGenerator(self):
            if self.use_GPU is False:
                g = PytorchDescriptorGenerator(**self.expected_params)
                g_pickled = pickle.dumps(g, -1)
                g2 = pickle.loads(g_pickled)

                from smqtk.representation.descriptor_element_factory import DescriptorElementFactory
                from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
                lenna_elem = from_uri(self.lenna_image_fp)
                factory = DescriptorElementFactory(DescriptorMemoryElement, {})
                d = g.compute_descriptor(lenna_elem, factory).vector()
                d2 = g2.compute_descriptor(lenna_elem, factory).vector()
                numpy.testing.assert_allclose(d, d2, atol=1e-8)
            else:
                pass


        def test_invalid_datatype(self):
            kwargs = {'model_cls_name': 'test', 'model_uri':None}
            self.assertRaises(
                KeyError,
                PytorchDescriptorGenerator,
                **kwargs
            )


        @mock.patch('smqtk.algorithms.descriptor_generator.caffe_descriptor'
                    '.CaffeDescriptorGenerator._setup_network')
        def test_no_internal_compute_descriptor(self, m_cdg_setupNetwork):
            # This implementation's descriptor computation logic sits in async
            # method override due to caffe's natural multi-element computation
            # interface. Thus, ``_compute_descriptor`` should not be
            # implemented.

            # dummy network setup because _setup_network is mocked out
            g = PytorchDescriptorGenerator(**self.expected_params)
            self.assertRaises(
                NotImplementedError,
                g._compute_descriptor, None
            )


        def test_compare_descriptors(self):
            # Compare the extracted feature is equal to the one
            # extracted directly from the model.
            d = PytorchDescriptorGenerator(**self.expected_params)
            lenna_elem = from_uri(self.lenna_image_fp)
            descr = d.compute_descriptor(lenna_elem).vector()

            from PIL import Image
            from torch.autograd import Variable
            img = Image.open(self.lenna_image_fp)
            img = img.resize((256, 256), Image.BILINEAR).convert('RGB')

            from smqtk.pytorch_model import get_pytorchmodel_element_impls
            pt_model = get_pytorchmodel_element_impls()[self.model_cls_name]()

            self.model_cls = pt_model.model_def()
            self.transform = pt_model.transforms()
            self.model_cls.eval()

            img = self.transform(img)
            if self.use_GPU:
                img = img.cuda()
                self.model_cls = self.model_cls.cuda()

            (expected_descr, _) = self.model_cls(Variable(img.unsqueeze(0)))

            if self.use_GPU:
                expected_descr = expected_descr.data.cpu().squeeze().numpy()
            else:
                expected_descr = expected_descr.data.squeeze().numpy()
            numpy.testing.assert_allclose(descr, expected_descr, atol=1e-8)


        def test_compute_descriptor_async_no_data(self):
            # Should get a ValueError when given no descriptors to async method
            g = PytorchDescriptorGenerator(**self.expected_params)
            self.assertRaises(
                ValueError,
                g.compute_descriptor_async, []
            )
