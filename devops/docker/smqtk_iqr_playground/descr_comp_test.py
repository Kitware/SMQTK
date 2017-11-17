"""
Test computing a descriptor on something.
"""

from smqtk.algorithms.descriptor_generator.caffe_descriptor import CaffeDescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement

e = DataFileElement("/home/smqtk/smqtk/source/python/smqtk/tests/data/Lenna.png")

gen = CaffeDescriptorGenerator(
    "/home/smqtk/caffe/source/models/bvlc_alexnet/deploy.prototxt",
    "/home/smqtk/caffe/source/models/bvlc_alexnet/bvlc_alexnet.caffemodel",
    "/home/smqtk/caffe/source/data/ilsvrc12/imagenet_mean.binaryproto",
    use_gpu=False, load_truncated_images=True
)

d = gen.compute_descriptor(e)

assert d.vector() is not None
