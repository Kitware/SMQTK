"""
Train and generate models for the SMQTK IQR Application.

This application takes the same configuration file as the IqrService REST
service.  To generate a default configuration, please refer to the
``runApplication`` tool for the ``IqrService`` application:

    runApplication -a IqrService -g config.IqrService.json
"""
import argparse
import glob
import json
import logging
import os.path as osp

import six

from smqtk import algorithms
from smqtk import representation
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import bin_utils, jsmin, plugin
from smqtk.web.iqr_service import IqrService


__author__ = 'paul.tunison@kitware.com'


def cli_parser():
    # Forgoing the ``bin_utils.basic_cli_parser`` due to our use of dual
    # configuration files for this utility.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('-v', '--verbose',
                        default=False, action='store_true',
                        help='Output additional debug logging.')
    parser.add_argument('-c', '--config',
                        metavar="PATH", nargs=2, required=True,
                        help='Path to the JSON configuration files. The first '
                             'file provided should be the configuration file '
                             'for the ``IqrSearchDispatcher`` web-application '
                             'and the second should be the configuration file '
                             'for the ``IqrService`` web-application.')

    parser.add_argument("-t", "--tab",
                        default=None, required=True,
                        help="The configuration \"tab\" of the "
                             "``IqrSearchDispatcher`` configuration to use. "
                             "This informs what dataset to add the input data "
                             "files to.")
    parser.add_argument("input_files",
                        metavar='GLOB', nargs="+",
                        help="Shell glob to files to add to the configured "
                             "data set.")

    return parser


def main():
    args = cli_parser().parse_args()

    ui_config_filepath, iqr_config_filepath = args.config
    llevel = logging.DEBUG if args.verbose else logging.INFO
    tab = args.tab
    input_files_globs = args.input_files

    # Not using `bin_utils.utility_main_helper`` due to deviating from single-
    # config-with-default usage.
    bin_utils.initialize_logging(logging.getLogger('smqtk'), llevel)
    bin_utils.initialize_logging(logging.getLogger('__main__'), llevel)
    log = logging.getLogger(__name__)

    log.info("Loading UI config: '{}'".format(ui_config_filepath))
    ui_config, ui_config_loaded = bin_utils.load_config(ui_config_filepath)
    log.info("Loading IQR config: '{}'".format(iqr_config_filepath))
    iqr_config, iqr_config_loaded = bin_utils.load_config(iqr_config_filepath)
    if not (ui_config_loaded and iqr_config_loaded):
        raise RuntimeError("One or both configuration files failed to load.")

    # Ensure the given "tab" exists in UI configuration.
    if tab is None:
        log.error("No configuration tab provided to drive model generation.")
        exit(1)
    if tab not in ui_config["iqr_tabs"]:
        log.error("Invalid tab provided: '{}'. Available tags: {}"
                  .format(tab, list(ui_config["iqr_tabs"])))
        exit(1)

    #
    # Gather Configurations
    #
    log.info("Extracting plugin configurations")

    ui_tab_config = ui_config["iqr_tabs"][tab]
    iqr_plugins_config = iqr_config['iqr_service']['plugins']

    # Configure DataSet implementation and parameters
    data_set_config = ui_tab_config['data_set']

    # Configure DescriptorElementFactory instance, which defines what
    # implementation of DescriptorElement to use for storing generated
    # descriptor vectors below.
    descriptor_elem_factory_config = iqr_plugins_config['descriptor_factory']

    # Configure DescriptorGenerator algorithm implementation, parameters and
    # persistent model component locations (if implementation has any).
    descriptor_generator_config = iqr_plugins_config['descriptor_generator']

    # Configure NearestNeighborIndex algorithm implementation, parameters and
    # persistent model component locations (if implementation has any).
    nn_index_config = iqr_plugins_config['neighbor_index']

    #
    # Initialize data/algorithms
    #
    # Constructing appropriate data structures and algorithms, needed for the
    # IQR demo application, in preparation for model training.
    #
    log.info("Instantiating plugins")
    #: :type: representation.DataSet
    data_set = \
        plugin.from_plugin_config(data_set_config,
                                  representation.get_data_set_impls())
    descriptor_elem_factory = \
        representation.DescriptorElementFactory \
        .from_config(descriptor_elem_factory_config)
    #: :type: algorithms.DescriptorGenerator
    descriptor_generator = \
        plugin.from_plugin_config(descriptor_generator_config,
                                  algorithms.get_descriptor_generator_impls())
    #: :type: algorithms.NearestNeighborsIndex
    nn_index = \
        plugin.from_plugin_config(nn_index_config,
                                  algorithms.get_nn_index_impls())

    #
    # Build models
    #
    log.info("Adding files to dataset '{}'".format(data_set))
    for g in input_files_globs:
        g = osp.expanduser(g)
        if osp.isfile(g):
            data_set.add_data(DataFileElement(g, readonly=True))
        else:
            log.debug("Expanding glob: %s" % g)
            for fp in glob.iglob(g):
                data_set.add_data(DataFileElement(fp, readonly=True))

    # Generate a model if the generator defines a known generation method.
    try:
        log.debug("descriptor generator as model to generate?")
        descriptor_generator.generate_model(data_set)
    except AttributeError as ex:
        log.debug("descriptor generator as model to generate - Nope: {}"
                  .format(str(ex)))

    # Generate descriptors of data for building NN index.
    log.info("Computing descriptors for data set with {}"
             .format(descriptor_generator))
    data2descriptor = descriptor_generator.compute_descriptor_async(
        data_set, descriptor_elem_factory
    )

    # Possible additional support steps before building NNIndex
    try:
        # Fit the LSH index functor
        log.debug("Has LSH Functor to fit?")
        nn_index.lsh_functor.fit(six.itervalues(data2descriptor))
    except AttributeError as ex:
        log.debug("Has LSH Functor to fit - Nope: {}".format(str(ex)))

    log.info("Building nearest neighbors index {}".format(nn_index))
    nn_index.build_index(six.itervalues(data2descriptor))


if __name__ == "__main__":
    main()
