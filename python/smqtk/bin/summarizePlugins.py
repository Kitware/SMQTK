"""
Script to try importing available plugins.
Plugins that have issues will have a change to emmit warnings or errors here.

:author: paul.tunison@kitware.com

"""
import argparse
import json
import logging

import smqtk.algorithms
import smqtk.algorithms.nn_index.hash_index
import smqtk.algorithms.nn_index.lsh.functors
import smqtk.representation
import smqtk.utils.bin_utils
import smqtk.utils.plugin


def cli():
    description = "Print out information about what plugins are currently " \
                  "usable and the documentation headers for each " \
                  "implementation."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-v", "--verbose",
                        default=False, action="store_true",
                        help="Output debugging options as well.")
    parser.add_argument("--defaults",
                        default=False, type=str,
                        help="Optionally generate default configuration blocks "
                             "for each plugin structure and output as JSON to "
                             "the specified path.")

    return parser


def format_cls_description(name, cls):
    return "%s\n%s" % (name, cls)


def main():
    args = cli().parse_args()

    llevel = logging.INFO
    if args.verbose:
        llevel = logging.DEBUG
    smqtk.utils.bin_utils.initialize_logging(logging.getLogger("smqtk"), llevel)

    collect_defaults = args.defaults
    defaults = {}

    def collect_configs(name, impl_map):
        """
        :type name: str
        :type impl_map: dict
        """
        if collect_defaults:
            defaults[name] = smqtk.utils.plugin.make_config(impl_map)

    log = logging.getLogger("smqtk.checkPlugins")

    # Key is the interface type name
    plugin_info = {}
    # List of plugin_info keys in order they were added
    plugin_type_list = []

    #
    # smqtk.representation
    #
    log.info("Checking DataElement plugins")
    plugin_type_list.append("DataElement")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_data_element_impls()
    collect_configs('DataElement',
                    smqtk.representation.get_data_element_impls())

    log.info("Checking DataSet plugins")
    plugin_type_list.append("DataSet")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_data_set_impls()
    collect_configs('DataSet',
                    smqtk.representation.get_data_set_impls())

    log.info("Checking DescriptorElement plugins")
    plugin_type_list.append("DescriptorElement")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_descriptor_element_impls()
    collect_configs('DescriptorElement',
                    smqtk.representation.get_descriptor_element_impls())

    log.info("Checking DescriptorIndex plugins")
    plugin_type_list.append("DescriptorIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_descriptor_index_impls()
    collect_configs('DescriptorIndex',
                    smqtk.representation.get_descriptor_index_impls())

    #
    # smqtk.algorithms
    #
    log.info("Checking Classifier plugins")
    plugin_type_list.append('Classifier')
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_classifier_impls()
    collect_configs('Classifier',
                    smqtk.algorithms.get_classifier_impls())

    log.info("Checking DescriptorGenerator plugins")
    plugin_type_list.append("DescriptorGenerator")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_descriptor_generator_impls()
    collect_configs('DescriptorGenerator',
                    smqtk.algorithms.get_descriptor_generator_impls())

    log.info("Checking HashIndex plugins")
    plugin_type_list.append("HashIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.nn_index.hash_index.get_hash_index_impls()
    collect_configs('HashIndex',
                    smqtk.algorithms.nn_index.hash_index.get_hash_index_impls())

    log.info("Checking LshFunctor plugins")
    plugin_type_list.append("LshFunctor")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.nn_index.lsh.functors.get_lsh_functor_impls()
    collect_configs('LshFunctor',
                    smqtk.algorithms.nn_index.lsh.functors
                         .get_lsh_functor_impls())

    log.info("Checking NearestNeighborIndex plugins")
    plugin_type_list.append("NearestNeighborIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_nn_index_impls()
    collect_configs('NearestNeighborIndex',
                    smqtk.algorithms.get_nn_index_impls())

    log.info("Checking RelevancyIndex plugins")
    plugin_type_list.append("RelevancyIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_relevancy_index_impls()
    collect_configs('RelevancyIndex',
                    smqtk.algorithms.get_relevancy_index_impls())

    #
    # Print-out
    #
    print
    print
    for k in plugin_type_list:
        print "[Type]", k
        print '='*(7+len(k))
        print
        for l, t in plugin_info[k].items():
            print ":: "+l
            if t.__doc__:
                print t.__doc__.rstrip()
                print
        print
        print

    if collect_defaults:
        with open(collect_defaults, 'w') as f:
            json.dump(defaults, f, indent=4, sort_keys=True)
        log.info("Wrote default configuration dictionaries to: %s",
                 collect_defaults)


if __name__ == "__main__":
    main()
