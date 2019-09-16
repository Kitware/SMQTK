"""
Test computing a descriptor on something.
"""

import os
from smqtk.algorithms.descriptor_generator.caffe_descriptor \
    import CaffeDescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement


data_filepath = "/usr/local/lib/python2.7/dist-packages/smqtk/tests/data/" \
                "Lenna.png"
assert os.path.isfile(data_filepath)
e = DataFileElement(data_filepath)

gen = CaffeDescriptorGenerator(
    DataFileElement("/home/smqtk/caffe/msra_resnet/ResNet-50-deploy.prototxt"),
    DataFileElement("/home/smqtk/caffe/msra_resnet/ResNet-50-model.caffemodel"),
    DataFileElement("/home/smqtk/caffe/msra_resnet/ResNet_mean.binaryproto"),
    return_layer="pool5",
    use_gpu=True, load_truncated_images=True
)

# Uses default DescriptorMemoryElement factory.
d = gen.generate_one_element(e)

assert d.vector() is not None
