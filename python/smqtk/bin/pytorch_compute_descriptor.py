import logging
from smqtk.utils.bin_utils import initialize_logging
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
import sys
sys.path.extend(['/home/bdong/XAI/caffe-1.0/python'])
from smqtk.algorithms.nn_index.flann import FlannNearestNeighborsIndex

# Import some butterfly data
files = ["/home/bdong/XAI/leedsbutterfly/images/001{:04d}.png".format(i) for i in range(4,10)]
from smqtk.representation.data_element.file_element import DataFileElement
el = [DataFileElement(d) for d in files]

initialize_logging(logging.getLogger('__main__'), logging.DEBUG)

# Create a model.  This assumes that you have properly set up a proper Caffe environment for SMQTK

cd = get_descriptor_generator_impls()['PytorchDescriptorGenerator'](
        # model_cls = Siamese(),
        model_cls_name = 'Siamese',
        model_uri = "/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/non_itar_siamese/snapshot_epoch_10.pt",
        # model_cls = models.resnet50(pretrained=True),
        # model_uri = None,
        resize_val = 224,
        batch_size = 10,
        use_gpu = True,
        in_gpu_device_id = None)

# Set up a factory for our vector (here in-memory storage)
from smqtk.representation.descriptor_element_factory import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
factory = DescriptorElementFactory(DescriptorMemoryElement, {})

# Compute features on the first image
descriptors = []
for item in el:
    d = cd.compute_descriptor(item, factory)
    descriptors.append(d)


index = FlannNearestNeighborsIndex(distance_method="euclidean",
                                   random_seed=42, index_uri="nn.index",
                                   parameters_uri="nn.params",
                                   descriptor_cache_uri="nn.cache")
index.build_index(descriptors)
