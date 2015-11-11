#!/usr/bin/env python
"""
Script to try importing available plugins.
Plugins that have issues will have a change to emmit warnings or errors here.

:author: paul.tunison@kitware.com

"""
import logging

import smqtk.algorithms
import smqtk.representation
import smqtk.utils.bin_utils


def cli():
    parser = smqtk.utils.bin_utils.SMQTKOptParser()

    parser.add_option("-v", "--verbose",
                      default=False, action="store_true",
                      help="Output debugging options as well.")

    return parser.parse_args()


def format_cls_description(name, cls):
    return "%s\n%s" % (name, cls)


def main():
    opts, args = cli()

    llevel = logging.INFO
    if opts.verbose:
        llevel = logging.DEBUG
    smqtk.utils.bin_utils.initialize_logging(logging.getLogger("smqtk"), llevel)

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

    log.info("Checking DataSet plugins")
    plugin_type_list.append("DataSet")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_data_set_impls()

    log.info("Checking DescriptorElement plugins")
    plugin_type_list.append("DescriptorElement")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_descriptor_element_impls()

    log.info("Checking DescriptorIndex plugins")
    plugin_type_list.append("DescriptorIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_descriptor_index_impls()

    log.info("Checking CodeIndex plugins")
    plugin_type_list.append("CodeIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.representation.get_code_index_impls()

    #
    # smqtk.algorithms
    #
    log.info("Checking DescriptorGenerator plugins")
    plugin_type_list.append("DescriptorGenerator")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_descriptor_generator_impls()

    log.info("Checking NearestNeighborIndex plugins")
    plugin_type_list.append("NearestNeighborIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_nn_index_impls()

    log.info("Checking RelevancyIndex plugins")
    plugin_type_list.append("RelevancyIndex")
    plugin_info[plugin_type_list[-1]] = \
        smqtk.algorithms.get_relevancy_index_impls()

    #
    # Print-out
    #
    print
    for k in plugin_type_list:
        print "[Type]", k
        print '='*(7+len(k))
        print
        for l, t in plugin_info[k].items():
            print ":: "+l
            if t.__doc__:
                print t.__doc__
        print


if __name__ == "__main__":
    main()
