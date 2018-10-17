"""
Tool for training the ITQ functor algorithm's model on descriptors in an
index.

By default, we use all descriptors in the configured index
(``uuids_list_filepath`` is not given a value).

The ``uuids_list_filepath`` configuration property is optional and should
be used to specify a sub-set of descriptors in the configured index to
train on. This only works if the stored descriptors' UUID is a type of
string.

The ``max_descriptors'' configuration property is optional and can be
used to cap the number of descriptors used to train the model.  If
more descriptors are available than requested, they are randomly
subsampled.
"""

import logging
import os.path
import random

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
        "max_descriptors": None,
    }


def cli_parser():
    return bin_utils.basic_cli_parser(__doc__)


def subsample(it, x, length):
    """Given an iterable it that has length length, return an iterable
    that consumes it and produces x elements by taking a random sample
    from it.

    """
    for elem in it:
        if random.random() * length < x:
            yield elem
            x -= 1
        length -= 1


def main():
    args = cli_parser().parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    uuids_list_filepath = config['uuids_list_filepath']
    max_descriptors = config['max_descriptors']

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
        uuids = uuids_iter()
        log.info("Loading UUIDs list from file: %s", uuids_list_filepath)
        if max_descriptors:
            uuids = list(uuids)
            if max_descriptors < len(uuids):
                log.info("Subsampling UUIDs (old count=%d, new count=%d)",
                         len(uuids), max_descriptors)
                uuids = random.sample(uuids, max_descriptors)
        d_iter = descriptor_index.get_many_descriptors(uuids)
    else:
        d_length = len(descriptor_index)
        log.info("Using UUIDs from loaded DescriptorIndex (count=%d)",
                 d_length)
        if max_descriptors and max_descriptors < d_length:
            log.info("Subsampling loaded DescriptorIndex (new count=%d)",
                     max_descriptors)
            d_iter = subsample(descriptor_index, max_descriptors, d_length)
        else:
            d_iter = descriptor_index

    log.info("Fitting ITQ model")
    functor.fit(d_iter)
    log.info("Done")


if __name__ == '__main__':
    main()
