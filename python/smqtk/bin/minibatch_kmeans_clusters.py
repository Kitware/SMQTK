"""
Script for generating clusters from descriptors in a given descriptor set using
the mini-batch KMeans implementation from Scikit-learn
(http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).

By the nature of Scikit-learn's MiniBatchKMeans implementation, euclidean
distance is used to measure distance between descriptors.
"""
import logging
import os

import numpy
from six.moves import cPickle
from sklearn.cluster import MiniBatchKMeans

from smqtk.compute_functions import mb_kmeans_build_apply
from smqtk.representation import DescriptorSet
from smqtk.utils.cli import utility_main_helper, basic_cli_parser
from smqtk.utils.configuration import (
    Configurable,
    from_config_dict,
    make_default_config
)
from smqtk.utils.file import safe_create_dir


def default_config():

    # Trick for mixing in our Configurable class API on top of scikit-learn's
    # MiniBatchKMeans class in order to introspect construction parameters.
    # We never construct this class so we do not need to implement "pure
    # virtual" instance methods.
    # noinspection PyAbstractClass
    class MBKTemp (MiniBatchKMeans, Configurable):
        pass

    c = {
        "minibatch_kmeans_params": MBKTemp.get_default_config(),
        "descriptor_set": make_default_config(DescriptorSet.get_impls()),
        # Number of descriptors to run an initial fit with. This brings the
        # advantage of choosing a best initialization point from multiple.
        "initial_fit_size": 0,
        # Path to save generated KMeans centroids
        "centroids_output_filepath_npy": "centroids.npy"
    }

    # Change/Remove some KMeans params for more appropriate defaults
    del c['minibatch_kmeans_params']['compute_labels']
    del c['minibatch_kmeans_params']['verbose']
    c['minibatch_kmeans_params']['random_state'] = 0

    return c


def cli_parser():
    p = basic_cli_parser(__doc__)

    g_output = p.add_argument_group("output")
    g_output.add_argument('-o', '--output-map',
                          metavar="PATH",
                          help="Path to output the clustering class mapping "
                               "to. Saved as a pickle file with -1 format.")

    return p


def main():
    args = cli_parser().parse_args()
    config = utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    output_filepath = args.output_map
    if not output_filepath:
        raise ValueError("No path given for output map file (pickle).")

    #: :type: smqtk.representation.DescriptorSet
    descr_set = from_config_dict(config['descriptor_set'],
                                 DescriptorSet.get_impls())
    mbkm = MiniBatchKMeans(verbose=args.verbose,
                           compute_labels=False,
                           **config['minibatch_kmeans_params'])
    initial_fit_size = int(config['initial_fit_size'])

    d_classes = mb_kmeans_build_apply(descr_set, mbkm, initial_fit_size)

    log.info("Saving KMeans centroids to: %s",
             config['centroids_output_filepath_npy'])
    numpy.save(config['centroids_output_filepath_npy'], mbkm.cluster_centers_)

    log.info("Saving result classification map to: %s", output_filepath)
    safe_create_dir(os.path.dirname(output_filepath))
    with open(output_filepath, 'w') as f:
        cPickle.dump(d_classes, f, -1)

    log.info("Done")


if __name__ == '__main__':
    main()
