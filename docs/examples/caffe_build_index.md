The following is a concrete example of performing nearest neiborhood computation
using a set of ten butterfly images. Following examples is tested using
caffe (version rc2, http://caffe.berkeleyvision.org/install_apt.html) and may
work with master version of caffe from github.

Run following scripts to generate image_mean_filepath, network_model_filepath,
and network_prototxt_filepath

> caffe_src/ilsvrc12/get_ilsvrc_aux.sh
> carffe_src/scripts/download_model_binary.py  ./models/bvlc_reference_caffenet/

```python
from smqtk.algorithms.nn_index.flann import FlannNearestNeighborsIndex

# Import some butterfly data
urls = ["http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/examples/{:03d}.jpg".format(i) for i in range(1,11)]
from smqtk.representation.data_element.url_element import DataUrlElement
el = [DataUrlElement(d) for d in urls]

# Create a model. This assumes you have set up the colordescriptor executable.
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
cd = get_descriptor_generator_impls()['CaffeDescriptorGenerator'](
        network_prototxt_filepath="caffe/models/bvlc_reference_caffenet/deploy.prototxt",
        network_model_filepath="caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel",
        image_mean_filepath="caffe/data/ilsvrc12/imagenet_mean.binaryproto",
        return_layer="fc7",
        batch_size=1,
        use_gpu=False,
        gpu_device_id=0,
        network_is_bgr=True,
        data_layer="data",
        load_truncated_images=True)

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
                                   random_seed=42, index_filepath="nn.index",
                                   parameters_filepath="nn.params",
                                   descriptor_cache_filepath="nn.cache")
index.build_index(descriptors)
```
