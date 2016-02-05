#!/usr/bin/env python
"""
Script for training ITQ LSH functor model from input descriptors.
"""
import logging
import os.path

import smqtk.algorithms.nn_index.lsh.functors.itq
import smqtk.representation
import smqtk.utils.bin_utils
import smqtk.utils.plugin


__author__ = "paul.tunison@kitware.com"


def default_config():
    return {
        "itq_config":
            smqtk.algorithms.nn_index.lsh.functors.itq.ItqFunctor
            .get_default_config(),
        "uuids_list_filepath": None,
        "descriptor_index": smqtk.utils.plugin.make_config(
            smqtk.representation.get_descriptor_index_impls
        ),
        "parallel": {
            "index_load_cores": 2,
            "use_multiprocessing": True,
        },
    }


def main():
    description = """
    Tool for training the ITQ functor algorithm's model on descriptors in an
    index.

    By default, we use all descriptors in the configured index
    (``uuids_list_filepath`` is not given a value).

    The ``uuids_list_filepath`` configuration property is optional and should
    be used to specify a sub-set of descriptors in the configured index to
    train on. This only works if the stored descriptors' UUID is a type of
    string.
    """
    args, config = smqtk.utils.bin_utils.utility_main_helper(default_config(),
                                                             description)
    log = logging.getLogger(__name__)

    uuids_list_filepath = config['uuids_list_filepath']
    p_index_load_cores = config['parllel']['index_load_cores']
    p_use_multiprocessing = config['parallel']['use_multiprocessing']

    log.info("Initializing ITQ functor")
    #: :type: smqtk.algorithms.nn_index.lsh.functors.itq.ItqFunctor
    functor = smqtk.algorithms.nn_index.lsh.functors.itq.ItqFunctor\
        .from_config(config['itq_config'])

    log.info("Initializing DescriptorIndex [type=%s]",
             config['descriptor_index']['type'])
    descriptor_index = smqtk.utils.plugin.from_plugin_config(
        config['descriptor_index'],
        smqtk.representation.get_descriptor_index_impls,
    )

    if uuids_list_filepath and os.path.isfile(uuids_list_filepath):
        log.info("Loading UUIDs list from file: %s", uuids_list_filepath)
        def uuids_iter():
            with

if __name__ == '__main__':
    main()
