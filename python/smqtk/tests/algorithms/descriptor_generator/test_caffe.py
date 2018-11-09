from __future__ import division, print_function
import inspect
import os
import pickle
import unittest

import PIL.Image
import mock
import numpy
import requests
from matplotlib.cbook import get_sample_data

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.algorithms.descriptor_generator.caffe_descriptor import \
    caffe, CaffeDescriptorGenerator
# Testing protected helper function
# noinspection PyProtectedMember
from smqtk.algorithms.descriptor_generator.caffe_descriptor import \
    _process_load_img_array
from smqtk.representation.data_element import from_uri
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.tests import TEST_DATA_DIR
from smqtk.utils.configuration import to_config_dict


def _data_element_from_url(url):
    """Helper function to avoid repeatedly downloading data in tests"""
    r = requests.get(url)
    r.raise_for_status()
    content_type = requests.get(url).headers['content-type']
    return DataMemoryElement(bytes=r.content, content_type=content_type)


if CaffeDescriptorGenerator.is_usable():

    class TestCaffeDesctriptorGenerator (unittest.TestCase):

        hopper_image_fp = get_sample_data('grace_hopper.png', asfileobj=False)
        hopper_alexnet_fc7_descr_fp = os.path.join(
            TEST_DATA_DIR, 'Hopper.alexnet_fc7_output.npy'
        )

        # Dummy Caffe configuration files + weights
        # - weights is actually an empty file (0 bytes), which caffe treats
        #   as random/zero values (not sure exactly what's happening, but
        #   always results in a zero-vector).
        dummy_net_topo_elem = DataFileElement(
            os.path.join(TEST_DATA_DIR, 'caffe.dummpy_network.prototxt'),
            readonly=True
        )
        dummy_caffe_model_elem = DataFileElement(
            os.path.join(TEST_DATA_DIR, 'caffe.empty_model.caffemodel'),
            readonly=True
        )
        dummy_img_mean_elem = DataFileElement(
            os.path.join(TEST_DATA_DIR, 'caffe.dummy_mean.npy'),
            readonly=True
        )

        @classmethod
        def setup_class(cls):
            cls.alexnet_prototxt_elem = _data_element_from_url(
                'https://data.kitware.com/api/v1/file/57e2f3fd8d777f10f26e532c'
                '/download'
            )
            cls.alexnet_caffemodel_elem = _data_element_from_url(
                'https://data.kitware.com/api/v1/file/57dae22f8d777f10f26a2a86'
                '/download'
            )
            cls.image_mean_proto_elem = _data_element_from_url(
                'https://data.kitware.com/api/v1/file/57dae0a88d777f10f26a2a82'
                '/download'
            )

        def test_impl_findable(self):
            self.assertIn(CaffeDescriptorGenerator,
                          DescriptorGenerator.get_impls())

        @mock.patch('smqtk.algorithms.descriptor_generator.caffe_descriptor'
                    '.CaffeDescriptorGenerator._setup_network')
        def test_get_config(self, _m_cdg_setupNetwork):
            # Mocking set_network so we don't have to worry about actually
            # initializing any caffe things for this test.
            expected_params = {
                'network_prototxt': DataMemoryElement(),
                'network_model': DataMemoryElement(),
                'image_mean': DataMemoryElement(),
                'return_layer': 'layer name',
                'batch_size': 777,
                'use_gpu': False,
                'gpu_device_id': 8,
                'network_is_bgr': False,
                'data_layer': 'data-other',
                'load_truncated_images': True,
                'pixel_rescale': (.2, .8),
                'input_scale': 1.5,
            }
            # make sure that we're considering all constructor parameter options
            expected_param_keys = \
                set(inspect.getargspec(CaffeDescriptorGenerator.__init__)
                           .args[1:])
            self.assertSetEqual(set(expected_params.keys()),
                                expected_param_keys)
            g = CaffeDescriptorGenerator(**expected_params)
            for key in ('network_prototxt', 'network_model', 'image_mean'):
                expected_params[key] = to_config_dict(expected_params[key])
            self.assertEqual(g.get_config(), expected_params)

        @mock.patch('smqtk.algorithms.descriptor_generator.caffe_descriptor'
                    '.CaffeDescriptorGenerator._setup_network')
        def test_pickle_save_restore(self, m_cdg_setupNetwork):
            # Mocking set_network so we don't have to worry about actually
            # initializing any caffe things for this test.
            expected_params = {
                'network_prototxt': DataMemoryElement(),
                'network_model': DataMemoryElement(),
                'image_mean': DataMemoryElement(),
                'return_layer': 'layer name',
                'batch_size': 777,
                'use_gpu': False,
                'gpu_device_id': 8,
                'network_is_bgr': False,
                'data_layer': 'data-other',
                'load_truncated_images': True,
                'pixel_rescale': (.2, .8),
                'input_scale': 1.5,
            }
            g = CaffeDescriptorGenerator(**expected_params)
            # Initialization sets up the network on construction.
            self.assertEqual(m_cdg_setupNetwork.call_count, 1)

            g_pickled = pickle.dumps(g, -1)
            g2 = pickle.loads(g_pickled)
            # Network should be setup for second class class just like in
            # initial construction.
            self.assertEqual(m_cdg_setupNetwork.call_count, 2)

            self.assertIsInstance(g2, CaffeDescriptorGenerator)
            self.assertEqual(g.get_config(), g2.get_config())

        @mock.patch('smqtk.algorithms.descriptor_generator.caffe_descriptor'
                    '.CaffeDescriptorGenerator._setup_network')
        def test_invalid_datatype(self, _m_cdg_setupNetwork):
            # Test that a data element with an incorrect content type raises an
            # exception.

            # Passing purposefully bag constructor parameters and ignoring
            # Caffe network setup (above mocking).
            # noinspection PyTypeChecker
            g = CaffeDescriptorGenerator(None, None, None)
            bad_element = from_uri(os.path.join(TEST_DATA_DIR, 'test_file.dat'))
            self.assertRaises(
                ValueError,
                g.compute_descriptor,
                bad_element
            )

        def test_process_load_img(self):
            # using image shape, meaning no transformation should occur
            test_data_layer = 'data'
            test_transformer = \
                caffe.io.Transformer({test_data_layer: (1, 3, 600, 512)})

            hopper_elem = from_uri(self.hopper_image_fp)
            a_expected = numpy.asarray(PIL.Image.open(self.hopper_image_fp),
                                       numpy.float32)
            a = _process_load_img_array((
                hopper_elem, test_transformer, test_data_layer, None, None
            ))
            numpy.testing.assert_allclose(a, a_expected)

        @mock.patch('smqtk.algorithms.descriptor_generator.caffe_descriptor'
                    '.CaffeDescriptorGenerator._setup_network')
        def test_no_internal_compute_descriptor(self, _m_cdg_setupNetwork):
            # This implementation's descriptor computation logic sits in async
            # method override due to caffe's natural multi-element computation
            # interface. Thus, ``_compute_descriptor`` should not be
            # implemented.

            # Passing purposefully bag constructor parameters and ignoring
            # Caffe network setup (above mocking).
            # noinspection PyTypeChecker
            g = CaffeDescriptorGenerator(0, 0, 0)
            self.assertRaises(
                NotImplementedError,
                g._compute_descriptor, None
            )

        def test_compute_descriptor_dummy_model(self):
            # Caffe dummy network interaction test Lenna image)

            # Construct network with an empty model just to see that our
            # interaction with the Caffe API is successful. We expect a
            # zero-valued descriptor vector.
            g = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                         self.dummy_caffe_model_elem,
                                         self.dummy_img_mean_elem,
                                         return_layer='fc', use_gpu=False)
            d = g.compute_descriptor(from_uri(self.hopper_image_fp))
            self.assertAlmostEqual(d.vector().sum(), 0., 12)

        @unittest.skipUnless(DataUrlElement.is_usable(),
                             "URL resolution not functional")
        def test_compute_descriptor_from_url_hopper_description(self):
            # Caffe AlexNet interaction test (Grace Hopper image)
            # This is a long test since it has to download data for remote URIs
            d = CaffeDescriptorGenerator(
                self.alexnet_prototxt_elem,
                self.alexnet_caffemodel_elem,
                self.image_mean_proto_elem,
                return_layer='fc7',
                use_gpu=False,
            )
            hopper_elem = from_uri(self.hopper_image_fp)
            expected_descr = numpy.load(self.hopper_alexnet_fc7_descr_fp)
            descr = d.compute_descriptor(hopper_elem).vector()
            numpy.testing.assert_allclose(descr, expected_descr, atol=1e-5)

        def test_compute_descriptor_async_no_data(self):
            # Should get a ValueError when given no descriptors to async method
            g = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                         self.dummy_caffe_model_elem,
                                         self.dummy_img_mean_elem,
                                         return_layer='fc', use_gpu=False)
            self.assertRaises(
                ValueError,
                g.compute_descriptor_async, []
            )
