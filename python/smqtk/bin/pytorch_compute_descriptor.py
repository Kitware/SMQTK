import logging
from smqtk.utils.bin_utils import initialize_logging
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
import sys
sys.path.extend(['/home/bdong/XAI/caffe-1.0/python'])
from smqtk.algorithms.nn_index.flann import FlannNearestNeighborsIndex
from smqtk.representation.data_set.file_set import DataFileSet
import os.path as osp
from smqtk.utils.preview_cache import PreviewCache
import PIL.Image
from collections import OrderedDict

# Import some butterfly data
files = ["/home/bdong/XAI/leedsbutterfly/images/001{:04d}.png".format(i) for i in range(4,6)]
from smqtk.representation.data_element.file_element import DataFileElement
el = [DataFileElement(d) for d in files]

initialize_logging(logging.getLogger('__main__'), logging.DEBUG)

# Create a model.  This assumes that you have properly set up a proper Caffe environment for SMQTK

cd = get_descriptor_generator_impls()['PytorchSaliencyDescriptorGenerator'](
        model_cls_name = 'ImageNet_ResNet50',
        model_uri = None,
        resize_val = 224,
        batch_size = 500,
        use_gpu = True,
        in_gpu_device_id = None)

# Set up a factory for our vector (here in-memory storage)
from smqtk.representation.descriptor_element_factory import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
factory = DescriptorElementFactory(DescriptorMemoryElement, {})

# Compute features on the first image
descriptors = []
for item in el:
    d = cd.compute_descriptor(item, factory, topk_label_list=[1,2])
    print(d.saliency_map())
    descriptors.append(d)


data_set = DataFileSet(root_directory='./sa_map')
de = data_set.get_data(descriptors[0].saliency_map()[1])

preview_cache = PreviewCache(osp.join('./', "previews"))
preview_path = preview_cache.get_preview_image(de)
img = PIL.Image.open(preview_path)
img.save('test.png')

index = FlannNearestNeighborsIndex(distance_method="euclidean",
                                   random_seed=42, index_uri="nn.index",
                                   parameters_uri="nn.params",
                                   descriptor_cache_uri="nn.cache")
index.build_index(descriptors)
