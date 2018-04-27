# coding=utf-8
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

    def collect_plugins(interface_t):
        """
        Record discoverable implementations for an interface class type.

        :param interface_t: Interface class type.
        :type interface_t: smqtk.utils.SmqtkObject

        """
        type_name = interface_t.__name__
        log.info("Checking %s plugins", type_name)
        plugin_type_list.append(type_name)
        impl_map = interface_t.get_impls()
        plugin_info[plugin_type_list[-1]] = impl_map
        collect_configs(type_name, impl_map)

    #
    # smqtk.representation
    #
    collect_plugins(smqtk.representation.DataElement)
    collect_plugins(smqtk.representation.DataSet)
    collect_plugins(smqtk.representation.DescriptorElement)
    collect_plugins(smqtk.representation.DescriptorIndex)
    collect_plugins(smqtk.representation.KeyValueStore)

    #
    # smqtk.algorithms
    #
    collect_plugins(smqtk.algorithms.Classifier)
    collect_plugins(smqtk.algorithms.DescriptorGenerator)
    collect_plugins(smqtk.algorithms.HashIndex)
    collect_plugins(smqtk.algorithms.LshFunctor)
    collect_plugins(smqtk.algorithms.NearestNeighborsIndex)
    collect_plugins(smqtk.algorithms.RelevancyIndex)

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
