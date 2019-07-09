"""
Script for computing the saliency map for a given image and model.

This script is for testing/developing saliency.py.

"""
import os
import pdb

import numpy as np
import scipy.misc
import PIL.Image
import logging

from smqtk.algorithms.descriptor_generator.caffe_descriptor import CaffeDescriptorGenerator
from smqtk.algorithms.relevancy_index.libsvm_hik import LibSvmHikRelevancyIndex
from smqtk.representation.data_element import from_uri

from smqtk.utils import saliency


__author__ = "alina.barnett@kitware.com"


def main():
    
    logging.basicConfig()
    #logging.getLogger('smqtk').setLevel(logging.DEBUG)
    # paths

    T_img_name = 'test_img'
    in_img_path = '/home/local/KHQ/alina.barnett/AlinaCode/imgs/test_imgs/' + T_img_name + '.jpg'
    out_img_path = '/home/local/KHQ/alina.barnett/AlinaCode/imgs/sa_imgs/' + T_img_name + '_sa_SVM_fast.jpg'
    out_img_path_Bo = '/home/local/KHQ/alina.barnett/AlinaCode/imgs/sa_imgs/' + T_img_name + '_sa_Bo.jpg'
    network_prototxt_uri = '/home/local/KHQ/alina.barnett/AlinaCode/models/caffe_ResNet50/ResNet-50-deploy.prototxt'
    network_model_uri = '/home/local/KHQ/alina.barnett/AlinaCode/models/caffe_ResNet50/ResNet-50-model.caffemodel'
    image_mean_uri = '/home/local/KHQ/alina.barnett/AlinaCode/models/caffe_ResNet50/ResNet_mean.binaryproto'
    pos_img_path = '/home/local/KHQ/alina.barnett/AlinaCode/imgs/test_imgs/test_img_flower.jpg'
    neg_img_path = '/home/local/KHQ/alina.barnett/AlinaCode/imgs/test_imgs/test_img_self.jpg'

    if not os.path.isfile(in_img_path):
        print("Given in_img_path did not point to a file at {}.".format(in_img_path))
    if out_img_path is None:
        print("Need a path to out_img.")

    
    print("Importing test image from file: %s", in_img_path)
    #T_img = np.array(PIL.Image.open(in_img_path)) #PIL imports as hwc
    T_img = PIL.Image.open(in_img_path)
    query_img = PIL.Image.open(pos_img_path)
    
    print("Setting up caffe model from files: {}, {}, {}.".format(network_prototxt_uri, network_model_uri, image_mean_uri))
    descriptor_generator = CaffeDescriptorGenerator(network_prototxt_uri, network_model_uri, image_mean_uri,
                 return_layer='pool5',
                 batch_size=40, use_gpu=True, gpu_device_id=2)
    
    relevancy_index = LibSvmHikRelevancyIndex()
    
    pos = [descriptor_generator.compute_descriptor(from_uri(pos_img_path))]
    neg = [descriptor_generator.compute_descriptor(from_uri(neg_img_path))]
    
    ADJs = (pos, neg)

    overlayed_img = saliency.generate_saliency_map_fast(T_img, descriptor_generator, relevancy_index, ADJs)
    
    #overlayed_img_Bo = saliency.generate_saliency_map_Bo(T_img, descriptor_generator, query_img)
    
    # Output the overlayed_img from SVM context
    print("Writing overlayed_img to file: {}".format(out_img_path))
    #overlayed_img = PIL.Image.fromarray(overlayed_img.astype(np.uint8))
    overlayed_img.save(out_img_path)
    
    # Output the overlayed_img
    #print("Writing overlayed_img to file: {}".format(out_img_path_Bo))
    #overlayed_img = PIL.Image.fromarray(overlayed_img.astype(np.uint8))
    #overlayed_img.save(out_img_path_Bo)

    print("Done")


if __name__ == '__main__':
    main()
