#!/usr/bin/env python

import cPickle
import collections
import logging
import os

import numpy
from sklearn.cluster import MiniBatchKMeans

from smqtk.representation.descriptor_index import get_descriptor_index_impls
from smqtk.utils import Configurable
from smqtk.utils.bin_utils import utility_main_helper, report_progress
from smqtk.utils.file_utils import safe_create_dir
from smqtk.utils.parallel import parallel_map
from smqtk.utils.plugin import make_config, from_plugin_config


def default_config():
    class MBKTemp (MiniBatchKMeans, Configurable):
        pass

    c = {
        "minibatch_kmeans_params": MBKTemp.get_default_config(),
        "descriptor_index": make_config(get_descriptor_index_impls()),
        # Number of descriptors to run an initial fit with. This brings
        # advantages like
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

    # Transfer verbosity setting to MiniBatchKMeans constructor args
    config['minibatch_kmeans_params']['verbose'] = args.verbose

    output_filepath = args.output_map
    if not output_filepath:
        raise ValueError("No path given for output map file (pickle).")

    #: :type: smqtk.representation.DescriptorIndex
    index = from_plugin_config(config['descriptor_index'],
                               get_descriptor_index_impls())
    k = MiniBatchKMeans(**config['minibatch_kmeans_params'])

    ifit_count = int(config['initial_fit_size'])
    ifit_completed = False
    d_deque = collections.deque()
    d_fitted = 0

    d_vector_iter = parallel_map(lambda d: d.vector(), index,
                                 name="vector-collector",
                                 use_multiprocessing=False)

    for i, v in enumerate(d_vector_iter):
        d_deque.append(v)

        if ifit_count and not ifit_completed:
            if len(d_deque) == ifit_count:
                log.info("Initial fit using %d descriptors", len(d_deque))
                k.fit(d_deque)
                d_fitted += len(d_deque)
                d_deque.clear()
                ifit_completed = True
        elif len(d_deque) == k.batch_size:
            log.info("Partial fit with batch size %d", len(d_deque))
            k.partial_fit(d_deque)
            d_fitted += len(d_deque)
            d_deque.clear()

    # Final fit with any remaining descriptors
    if d_deque:
        log.info("Final partial fit of size %d", len(d_deque))
        k.partial_fit(d_deque)
        d_fitted += len(d_deque)
        d_deque.clear()

    log.info("Computing descriptor classes with final KMeans model")
    k.verbose = False
    d_classes = collections.defaultdict(set)
    d_uv_iter = parallel_map(lambda d: (d.uuid(), d.vector()),
                             index,
                             use_multiprocessing=False,
                             name="uv-collector")
    d_uc_iter = parallel_map(lambda (u, v): (u, k.predict(v[numpy.newaxis, :])[0]),
                             d_uv_iter,
                             use_multiprocessing=False,
                             name="uc-collector")
    rps = [0] * 7
    for uuid, c in d_uc_iter:
        d_classes[c].add(uuid)
        report_progress(log.debug, rps, 1.)
    rps[1] -= 1
    report_progress(log.debug, rps, 0)

    log.info("Saving result classification map to: %s", output_filepath)
    safe_create_dir(os.path.dirname(output_filepath))
    with open(output_filepath, 'w') as f:
        cPickle.dump(d_classes, f, -1)


if __name__ == '__main__':
    main()
