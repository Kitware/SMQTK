"""
Test computing a descriptor on something.
"""
from smqtk.algorithms.descriptor_generator.caffe_descriptor \
    import CaffeDescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.url_element import DataUrlElement


grace_hopper_img_url = "https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg"
e = DataUrlElement(grace_hopper_img_url)

gen = CaffeDescriptorGenerator(
    DataFileElement("/home/smqtk/caffe/msra_resnet/ResNet-50-deploy.prototxt"),
    DataFileElement("/home/smqtk/caffe/msra_resnet/ResNet-50-model.caffemodel"),
    DataFileElement("/home/smqtk/caffe/msra_resnet/ResNet_mean.binaryproto"),
    return_layer="pool5",
    use_gpu=False, load_truncated_images=True
)

# Uses default DescriptorMemoryElement factory.
d = gen.generate_one_element(e)

assert d.vector() is not None
print(d.vector())
