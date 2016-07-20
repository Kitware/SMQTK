#!/usr/bin/env python

import cPickle
import logging
import os

from sklearn.cluster import MiniBatchKMeans

from smqtk.compute_functions import mb_kmeans_build_apply
from smqtk.representation.descriptor_index import get_descriptor_index_impls
from smqtk.utils import Configurable
from smqtk.utils.bin_utils import utility_main_helper
from smqtk.utils.file_utils import safe_create_dir
from smqtk.utils.plugin import make_config, from_plugin_config


def default_config():
    class MBKTemp (MiniBatchKMeans, Configurable):
        pass

    c = {
        "minibatch_kmeans_params": MBKTemp.get_default_config(),
        "descriptor_index": make_config(get_descriptor_index_impls()),
        # Number of descriptors to run an initial fit with. This brings the
        # advantage of choosing a best initialization point from multiple.
        "initial_fit_size": 0,
    }

    # Change/Remove some KMeans params for more appropriate defaults
    del c['minibatch_kmeans_params']['compute_labels']
    del c['minibatch_kmeans_params']['verbose']
    c['minibatch_kmeans_params']['random_state'] = 0

    return c


def extend_parser(p):
    """
    :type p: argparse.ArgumentParser
    :rtype: argparse.ArgumentParser
    """
    g_output = p.add_argument_group("output")
    g_output.add_argument('-o', '--output-map',
                          metavar="PATH",
                          help="Path to output the clustering class mapping "
                               "to. Saved as a pickle file with -1 format.")

    return p


def main():
    description = """
    Script for generating clusters from descriptors in a given index using the
    mini-batch KMeans implementation from Scikit-learn.

    By the nature of Scikit-learn's MiniBatchKMeans implementation, euclidean
    distance is used.
    """
    args, config = utility_main_helper(default_config, description,
                                       extend_parser)
    log = logging.getLogger(__name__)

    output_filepath = args.output_map
    if not output_filepath:
        raise ValueError("No path given for output map file (pickle).")

    #: :type: smqtk.representation.DescriptorIndex
    index = from_plugin_config(config['descriptor_index'],
                               get_descriptor_index_impls())
    mbkm = MiniBatchKMeans(verbose=args.verbose,
                           compute_labels=False,
                           **config['minibatch_kmeans_params'])
    initial_fit_size = int(config['initial_fit_size'])

    d_classes = mb_kmeans_build_apply(index, mbkm, initial_fit_size)

    log.info("Saving result classification map to: %s", output_filepath)
    safe_create_dir(os.path.dirname(output_filepath))
    with open(output_filepath, 'w') as f:
        cPickle.dump(d_classes, f, -1)

    log.info("Done")


if __name__ == '__main__':
    main()
