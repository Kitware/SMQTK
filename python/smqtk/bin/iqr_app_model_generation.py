"""
Train and generate models for the SMQTK IQR Application.
"""
import glob
import json
import logging
import argparse
import os.path as osp

from smqtk import algorithms
from smqtk import representation
from smqtk.utils import bin_utils, jsmin, plugin

__author__ = 'paul.tunison@kitware.com'


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-c", "--config",
                        required=True,
                        help="IQR application configuration file.")
    parser.add_argument("-t", "--tab",
                        type=int, default=0,
                        help="The configuration tab to generate the model for.")
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='Show debug logging.')

    parser.add_argument("input_files",
                        metavar='GLOB', nargs="*",
                        help="Shell glob to files to add to the configured "
                             "data set.")

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    #
    # Setup logging
    #
    if not logging.getLogger().handlers:
        if args.verbose:
            bin_utils.initialize_logging(logging.getLogger(), logging.DEBUG)
        else:
            bin_utils.initialize_logging(logging.getLogger(), logging.INFO)
    log = logging.getLogger("smqtk.scripts.iqr_app_model_generation")

    search_app_config = json.loads(jsmin.jsmin(open(args.config).read()))

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

    # base actions on a specific IQR tab configuration (choose index here)
    if args.tab <  0 or args.tab > (len(search_app_config["iqr_tabs"]) - 1):
        log.error("Invalid tab number provided.")
        exit(1)

    search_app_iqr_config = search_app_config["iqr_tabs"][args.tab]

    # Configure DataSet implementation and parameters
    data_set_config = search_app_iqr_config['data_set']

    # Configure DescriptorGenerator algorithm implementation, parameters and
    # persistant model component locations (if implementation has any).
    descriptor_generator_config = search_app_iqr_config['descr_generator']

    # Configure NearestNeighborIndex algorithm implementation, parameters and
    # persistant model component locations (if implementation has any).
    nn_index_config = search_app_iqr_config['nn_index']

    # Configure RelevancyIndex algorithm implementation, parameters and
    # persistant model component locations (if implementation has any).
    #
    # The LibSvmHikRelevancyIndex implementation doesn't actually build a persistant
    # model (or doesn't have to that is), but we're leaving this block here in
    # anticipation of other potential implementations in the future.
    #
    rel_index_config = search_app_iqr_config['rel_index_config']

    # Configure DescriptorElementFactory instance, which defines what implementation
    # of DescriptorElement to use for storing generated descriptor vectors below.
    descriptor_elem_factory_config = search_app_iqr_config['descriptor_factory']

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
                                  representation.get_data_set_impls())
    #: :type: algorithms.DescriptorGenerator
    descriptor_generator = \
        plugin.from_plugin_config(descriptor_generator_config,
                                  algorithms.get_descriptor_generator_impls())

    #: :type: algorithms.NearestNeighborsIndex
    nn_index = \
        plugin.from_plugin_config(nn_index_config,
                                  algorithms.get_nn_index_impls())

    #: :type: algorithms.RelevancyIndex
    rel_index = \
        plugin.from_plugin_config(rel_index_config,
                                  algorithms.get_relevancy_index_impls())

    #
    # Build models
    #
    # Perform the actual building of the models.
    #

    # Add data files to DataSet
    DataFileElement = representation.get_data_element_impls()["DataFileElement"]

    for fp in args.input_files:
        fp = osp.expanduser(fp)
        if osp.isfile(fp):
            data_set.add_data(DataFileElement(fp))
        else:
            log.debug("Expanding glob: %s" % fp)
            for g in glob.iglob(fp):
                data_set.add_data(DataFileElement(g))

    # Generate a mode if the generator defines a known generation method.
    if hasattr(descriptor_generator, "generate_model"):
        descriptor_generator.generate_model(data_set)
    # Add other if-else cases for other known implementation-specific generation
    # methods stubs

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


if __name__ == "__main__":
    main()
