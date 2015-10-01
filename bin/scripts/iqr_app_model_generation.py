import logging

from smqtk import algorithms
from smqtk import representation
from smqtk.utils import bin_utils, plugin


__author__ = 'paul.tunison@kitware.com'


#
# Setup logging
#
if not logging.getLogger().handlers:
    bin_utils.initialize_logging(logging.getLogger(), logging.INFO)


#
# Input parameters
#
data_set_config = {
    "type": "DataFileSet",
    "DataFileSet": {
        "root_directory": "/Users/purg/dev/smqtk/source/data/FileDataSets/example_image/data",
        "sha1_chunk": 10
    }
}

descriptor_elem_factory_config = {
    # 'DescriptorFileElement': {
    #     'save_dir': None,
    #     'subdir_split': None
    # },
    'DescriptorMemoryElement': {},
    # 'SolrDescriptorElement': {
    #     'commit_on_set': True,
    #     'persistent_connection': False,
    #     'solr_conn_addr': None,
    #     'timeout': 10,
    #     'timestamp_field': None,
    #     'type_field': None,
    #     'uuid_field': None,
    #     'vector_field': None
    # },
    'type': 'DescriptorMemoryElement'
}

descriptor_generator_config = {
    "ColorDescriptor_Image_csift": {
        "flann_autotune": False,
        "flann_sample_fraction": 0.75,
        "flann_target_precision": 0.95,
        "kmeans_k": 1024,
        "model_directory": "/Users/purg/dev/smqtk/source/data/ContentDescriptors/ColorDescriptor/csift/example_image",
        "random_seed": 42,
        "use_spatial_pyramid": False,
        "work_directory": "/Users/purg/dev/smqtk/source/work/ContentDescriptors/ColorDescriptor/csift/example_image"
    },
    "type": "ColorDescriptor_Image_csift"
}

nn_index_config = {
    # "FlannNearestNeighborsIndex": {
    #     "autotune": false,
    #     "descriptor_cache_filepath": "descriptor_cache.pickle",
    #     "distance_method": "hik",
    #     "index_filepath": "index.flann",
    #     "parameters_filepath": "index_parameters.pickle",
    #     "random_seed": null,
    #     "sample_fraction": 0.1,
    #     "target_precision": 0.95
    # },
    "ITQNearestNeighborsIndex": {
        "bit_length": 8,
        "code_index": {
            "MemoryCodeIndex": {
                "file_cache": "/Users/purg/dev/smqtk/source/data/"
                              "NearestNeighborsIndex/ITQNearestNeighborsIndex/"
                              "example_image/csift/mem_code_index.pickle"
            },
            "type": "MemoryCodeIndex"
        },
        "distance_method": "cosine",
        "itq_iterations": 50,
        "mean_vec_filepath": "/Users/purg/dev/smqtk/source/data/"
                             "NearestNeighborsIndex/ITQNearestNeighborsIndex/"
                             "example_image/csift/mean_vec.npy",
        "random_seed": 42,
        "rotation_filepath": "/Users/purg/dev/smqtk/source/data/"
                             "NearestNeighborsIndex/ITQNearestNeighborsIndex/"
                             "example_image/csift/rotation.npy"
    },
    "type": "ITQNearestNeighborsIndex"
}

rel_index_config = {
    "LibSvmHikRelevancyIndex": {
        "descr_cache_filepath": None
    },
    "type": "LibSvmHikRelevancyIndex"
}


#
# Initialize data/algorithms
#

descriptor_elem_factory = \
    representation.DescriptorElementFactory \
    .from_config(descriptor_elem_factory_config)

#: :type: representation.DataSet
data_set = \
    plugin.from_plugin_config(data_set_config,
                              representation.get_data_set_impls)
#: :type: algorithms.DescriptorGenerator
descriptor_generator = \
    plugin.from_plugin_config(descriptor_generator_config,
                              algorithms.get_descriptor_generator_impls)

#: :type: algorithms.NearestNeighborsIndex
nn_index = \
    plugin.from_plugin_config(nn_index_config,
                              algorithms.get_nn_index_impls)

#: :type: algorithms.RelevancyIndex
rel_index = \
    plugin.from_plugin_config(rel_index_config,
                              algorithms.get_relevancy_index_impls)

#
# Build models
#

# Generate a mode if the generator defines a known generation method.
if hasattr(descriptor_generator, "generate_model"):
    descriptor_generator.generate_model(data_set)

# Generate descriptors of data for building NN index.
data2descriptor = descriptor_generator.compute_descriptor_async(
    data_set, descriptor_elem_factory
)

try:
    nn_index.build_index(data2descriptor.itervalues())
except RuntimeError:
    # Already built model, so skipping this step
    pass

rel_index.build_index(data2descriptor.itervalues())
