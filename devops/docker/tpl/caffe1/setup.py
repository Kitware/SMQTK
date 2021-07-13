#!/usr/bin/env python
import setuptools


setuptools.setup(
    name="caffe",
    version="1.0",
    description="Install setup for Caffe 1.0 that didn't come with caffe.",
    packages=setuptools.find_packages(),
    package_data={
        'caffe': ['_caffe.so', 'imagenet/ilsvrc_2012_mean.npy'],
    },
    install_requires=["numpy", "scikit-image", "protobuf"],
)
