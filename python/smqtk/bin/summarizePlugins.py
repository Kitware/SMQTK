"""
Print out information about what plugins are currently usable and the
documentation headers for each implementation.
"""
from __future__ import print_function

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose",
                        default=False, action="store_true",
                        help="Output additional debug logging.")
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

    def collect_plugins(type_name, impl_getter_fn):
        log.info("Checking %s plugins", type_name)
        plugin_type_list.append(type_name)
        impl_map = impl_getter_fn()
        plugin_info[plugin_type_list[-1]] = impl_map
        collect_configs(type_name, impl_map)

    #
    # smqtk.representation
    #
    collect_plugins('DataElement',
                    smqtk.representation.get_data_element_impls)
    collect_plugins('DataSet',
                    smqtk.representation.get_data_set_impls)
    collect_plugins('DescriptorElement',
                    smqtk.representation.get_descriptor_element_impls)
    collect_plugins('DescriptorIndex',
                    smqtk.representation.get_descriptor_index_impls)
    collect_plugins('KeyValueStore',
                    smqtk.representation.get_key_value_store_impls)

    #
    # smqtk.algorithms
    #
    collect_plugins('Classifier',
                    smqtk.algorithms.get_classifier_impls)
    collect_plugins('DescriptorGenerator',
                    smqtk.algorithms.get_descriptor_generator_impls)
    collect_plugins('HashIndex',
                    smqtk.algorithms.get_hash_index_impls)
    collect_plugins('LshFunctor',
                    smqtk.algorithms.get_lsh_functor_impls)
    collect_plugins('NearestNeighborIndex',
                    smqtk.algorithms.get_nn_index_impls)
    collect_plugins('RelevancyIndex',
                    smqtk.algorithms.get_relevancy_index_impls)

    #
    # Print-out
    #
    print()
    print()
    for k in plugin_type_list:
        print("[Type]", k)
        print('=' * (7 + len(k)))
        print()
        for l, t in plugin_info[k].items():
            print(":: " + l)
            if t.__doc__:
                print(t.__doc__.rstrip())
                print()
        print()
        print()

    if collect_defaults:
        with open(collect_defaults, 'w') as f:
            json.dump(defaults, f, indent=4, sort_keys=True,
                      separators=(',', ': '))
        log.info("Wrote default configuration dictionaries to: %s",
                 collect_defaults)


if __name__ == "__main__":
    main()
