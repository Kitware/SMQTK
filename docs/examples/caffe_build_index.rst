Nearest Neighbor Computation with Caffe
---------------------------------------

The following is a concrete example of performing a nearest neighbor computation
using a set of ten butterfly images. This example has been tested using
`Caffe version rc2`_, ) and may work with the master version of Caffe from GitHub_.

.. _`Caffe version rc2`: http://caffe.berkeleyvision.org/
.. _GitHub: https://github.com/BVLC/caffe

To generate the required model files :file:`image_mean_filepath` and  :file:`network_model_filepath`,
run the following scripts::

    caffe_src/ilsvrc12/get_ilsvrc_aux.sh
    caffe_src/scripts/download_model_binary.py  ./models/bvlc_reference_caffenet/

Once this is done, the nearest neighbor index for the butterfly images can be built with the following
code:

.. code-block:: python

    from smqtk.algorithms.nn_index.flann import FlannNearestNeighborsIndex

    # Import some butterfly data
    urls = ["http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/examples/{:03d}.jpg".format(i) for i in range(1,11)]
    from smqtk.representation.data_element.url_element import DataUrlElement
    el = [DataUrlElement(d) for d in urls]

    # Create a model.  This assumes that you have properly set up a proper Caffe environment for SMQTK
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
