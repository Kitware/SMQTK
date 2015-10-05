"""
Summary
=======
Script intended to provide an example on how to train or generate models for
data structures and algorithm types required for the IQR demo web application's
function.

After successfully completing the execution of this script, the implementation
configurations used here will now reference valid models. When running the IQR
demo application, its configuration should mirror the configurations used here
in order to function properly.


TODO
====
In the future, this script should be updated to have a command-line interface
such that it takes in the same configuration JSON file that the IQR demo
application would (possibly with extra parameters for data file locations),
which would in turn drive structure/model generation.

"""
import glob
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
# The following dictionaries are JSON configurations that are used to
# configure the various data structures and algorithms needed for the IQR demo
# application. Values here can be changed to suit your specific data and
# algorithm needs.
#
# See algorithm implementation doc-strings for more information on configuration
# parameters (see implementation class ``__init__`` method).
#

# Shell glob for where input data is located.
input_image_file_glob = "/Users/purg/dev/smqtk/source/data/FileDataSets/" \
                        "example_image/images/*/*"

# Configure DataSet implementation and parameters
data_set_config = {
    "type": "DataFileSet",
    "DataFileSet": {
        "root_directory": "/Users/purg/dev/smqtk/source/data/FileDataSets/"
                          "example_image/data",
        "uuid_chunk": 10
    }
}

# Configure DescriptorElementFactory instance, which defines what implementation
# of DescriptorElement to use for storing generated descriptor vectors.
descriptor_elem_factory_config = {
    'DescriptorMemoryElement': {},
    # Some other descriptor element implementations:
    # 'DescriptorFileElement': {
    #     'save_dir': None,
    #     'subdir_split': None
    # },
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

# Configure DescriptorGenerator algorithm implementation, parameters and
# persistant model component locations (if implementation has any).
descriptor_generator_config = {
    "ColorDescriptor_Image_csift": {
        "flann_autotune": False,
        "flann_sample_fraction": 0.75,
        "flann_target_precision": 0.95,
        "kmeans_k": 1024,
        "model_directory": "/Users/purg/dev/smqtk/source/data/"
                           "ContentDescriptors/ColorDescriptor/csift/"
                           "example_image",
        "random_seed": 42,
        "use_spatial_pyramid": False,
        "work_directory": "/Users/purg/dev/smqtk/source/work/"
                          "ContentDescriptors/ColorDescriptor/csift/"
                          "example_image"
    },
    "type": "ColorDescriptor_Image_csift"
}

# Configure NearestNeighborIndex algorithm implementation, parameters and
# persistant model component locations (if implementation has any).
nn_index_config = {
    "ITQNearestNeighborsIndex": {
        "bit_length": 64,
        "code_index": {
            "MemoryCodeIndex": {
                "file_cache": "/Users/purg/dev/smqtk/source/data/"
                              "NearestNeighborsIndex/ITQNearestNeighborsIndex/"
                              "example_image/csift/mem_code_index.pickle"
            },
            "type": "MemoryCodeIndex"
        },
        "distance_method": "hik",
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

# Configure RelevancyIndex algorithm implementation, parameters and
# persistant model component locations (if implementation has any).
#
# The LibSvmHikRelevancyIndex implementation doesn't actually build a persistant
# model (or doesn't have to that is), but we're leaving this block here in
# anticipation of other potential implementations in the future.
#
rel_index_config = {
    "LibSvmHikRelevancyIndex": {
        "descr_cache_filepath": None,
        "autoneg_select_ratio": 1
    },
    "type": "LibSvmHikRelevancyIndex"
}


#
# Initialize data/algorithms
#
# Constructing appropriate data structures and algorithms, needed for the IQR
# demo application, in preparation for model training.
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
# Perform the actual building or the models.
#

# Add data files to DataSet
DataFileElement = representation.get_data_element_impls()["DataFileElement"]
data_set.add_data(*[DataFileElement(fp) for fp
                    in glob.iglob(input_image_file_glob)])

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
