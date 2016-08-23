"""
Tool for training the ITQ functor algorithm's model on descriptors in an
index.

By default, we use all descriptors in the configured index
(``uuids_list_filepath`` is not given a value).

The ``uuids_list_filepath`` configuration property is optional and should
be used to specify a sub-set of descriptors in the configured index to
train on. This only works if the stored descriptors' UUID is a type of
string.
"""

import logging
import os.path

from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.representation import (
    get_descriptor_index_impls,
)
from smqtk.utils import (
    bin_utils,
    plugin,
)


__author__ = "paul.tunison@kitware.com"


def default_config():
    return {
        "itq_config": ItqFunctor.get_default_config(),
        "uuids_list_filepath": None,
        "descriptor_index": plugin.make_config(get_descriptor_index_impls()),
    }


def cli_parser():
    return bin_utils.basic_cli_parser(__doc__)


def main():
    args = cli_parser().parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    uuids_list_filepath = config['uuids_list_filepath']

    log.info("Initializing ITQ functor")
    #: :type: smqtk.algorithms.nn_index.lsh.functors.itq.ItqFunctor
    functor = ItqFunctor.from_config(config['itq_config'])

    log.info("Initializing DescriptorIndex [type=%s]",
             config['descriptor_index']['type'])
    #: :type: smqtk.representation.DescriptorIndex
    descriptor_index = plugin.from_plugin_config(
        config['descriptor_index'],
        get_descriptor_index_impls(),
    )

    if uuids_list_filepath and os.path.isfile(uuids_list_filepath):
        def uuids_iter():
            with open(uuids_list_filepath) as f:
                for l in f:
                    yield l.strip()
        log.info("Loading UUIDs list from file: %s", uuids_list_filepath)
        d_iter = descriptor_index.get_many_descriptors(uuids_iter())
    else:
        log.info("Using UUIDs from loaded DescriptorIndex (count=%d)",
                 len(descriptor_index))
        d_iter = descriptor_index

    log.info("Fitting ITQ model")
    functor.fit(d_iter)
    log.info("Done")


if __name__ == '__main__':
    main()
