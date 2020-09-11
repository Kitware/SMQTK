"""
Tool for training the ITQ functor algorithm's model on descriptors in a
set.

By default, we use all descriptors in the configured set
(``uuids_list_filepath`` is not given a value).

The ``uuids_list_filepath`` configuration property is optional and should
be used to specify a sub-set of descriptors in the configured set to
train on. This only works if the stored descriptors' UUID is a type of
string.
"""

import logging
import os

from smqtk.algorithms.nn_index.lsh.functors.itq import ItqFunctor
from smqtk.representation import DescriptorSet
from smqtk.utils import (
    cli,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


__author__ = "paul.tunison@kitware.com"


def default_config():
    return {
        "itq_config": ItqFunctor.get_default_config(),
        "uuids_list_filepath": None,
        "descriptor_set": make_default_config(DescriptorSet.get_impls()),
    }


def cli_parser():
    return cli.basic_cli_parser(__doc__)


def main():
    args = cli_parser().parse_args()
    config = cli.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    uuids_list_filepath = config['uuids_list_filepath']

    log.info("Initializing ITQ functor")
    #: :type: smqtk.algorithms.nn_index.lsh.functors.itq.ItqFunctor
    functor = ItqFunctor.from_config(config['itq_config'])

    log.info("Initializing DescriptorSet [type=%s]",
             config['descriptor_set']['type'])
    #: :type: smqtk.representation.DescriptorSet
    descriptor_set = from_config_dict(
        config['descriptor_set'],
        DescriptorSet.get_impls(),
    )

    if uuids_list_filepath and os.path.isfile(uuids_list_filepath):
        def uuids_iter():
            with open(uuids_list_filepath) as f:
                for l in f:
                    yield l.strip()
        log.info("Loading UUIDs list from file: %s", uuids_list_filepath)
        d_iter = descriptor_set.get_many_descriptors(uuids_iter())
    else:
        log.info("Using UUIDs from loaded DescriptorSet (count=%d)",
                 len(descriptor_set))
        d_iter = descriptor_set

    log.info("Fitting ITQ model")
    functor.fit(d_iter)
    log.info("Done")


if __name__ == '__main__':
    main()
