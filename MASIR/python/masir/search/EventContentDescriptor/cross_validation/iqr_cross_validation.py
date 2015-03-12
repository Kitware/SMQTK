# coding=utf-8
"""

Attempt at implementing k-fold cross validation for model parameter tuning.

"""
__author__ = 'paul.tunison@kitware.com'

import logging
import numpy

from masir import FeatureMemory
from masir.search.EventContentDescriptor import iqr_modules
from masir.search.EventContentDescriptor.cross_validation import \
    perf_estimation


def masir_svm_cross_validate(k_folds, parameter_sets, fm,
                             positive_ids, negative_ids,
                             metric='ap_prime'):
    """
    Perform K-fold cross validation over the given data and positive set.

    Only uses one fold for testing per fold iteration (K1 CV).

    :param k_folds: Number of folds to perform
    :type k_folds: int
    :param parameter_sets: Iterable of parameter strings for libSVM
    :type parameter_sets: Iterable of str
    :param fm: Dataset feature memory object
    :type fm: masir.FeatureMemory.FeatureMemory
    :param positive_ids: Iterable of positive image IDs in given dataset
    :type positive_ids: list of int
    :param negative_ids: Iterable of negative image IDs in given dataset
    :type negative_ids: list of int
    :param metric: Average precision metric flavor to use. Must be one of:
        [ "ap_prime", "ap", "R0_star" ]
    :type metric: str

    :return: Optimal parameter set
    :rtype: str

    """
    log = logging.getLogger("masir_SVM_CV")

    # Input checks
    assert not set(positive_ids).intersection(negative_ids), \
        "Common IDs in positive and negative ID sets!"

    #
    # Partition the pos/neg IDs into k slices for positive and negative IDs
    #
    k_folds = int(k_folds)
    fold_index = range(k_folds)
    pos_partition_interval = len(positive_ids) / float(k_folds)
    neg_partition_interval = len(negative_ids) / float(k_folds)

    pos_fold_indices = [0]
    neg_fold_indices = [0]
    for f in range(1, k_folds):
        pos_fold_indices.append(int(pos_partition_interval * f))
        neg_fold_indices.append(int(neg_partition_interval * f))
    pos_fold_indices.append(len(positive_ids))
    neg_fold_indices.append(len(negative_ids))

    # iterables of ID set slices for each fold
    pos_fold_slices = tuple(slice(pos_fold_indices[f], pos_fold_indices[f + 1])
                            for f in fold_index)
    neg_fold_slices = tuple(slice(neg_fold_indices[f], neg_fold_indices[f + 1])
                            for f in fold_index)

    #+
    # DEBUG
    log.debug("Pos fold slices: %s", pos_fold_slices)
    log.debug("Neg fold slices: %s", neg_fold_slices)
    #-

    #
    # CV vars
    #

    # Collection of average CV k-fold precisions per parameter set tested
    #: :type: list of float
    p_set_avg_precision = []

    # Average precision metric flavor to use
    # must be one of: [ "ap_prime", "ap", "R0_star" ]
    metric = metric or "ap_prime"

    #
    # Train, test and score for each parameter set
    #
    for param_set in parameter_sets:
        # For each parameter set, train/test
        # For each fold, create an SVM model with training fold, test on testing
        #   fold, compute average precision
        msg = "===== Parameters: %s " % param_set
        log.info(msg + '=' * (80 - len(msg)))

        # List entries parallel to fold range
        #: :type: list of dict
        fold_results = []

        # f denotes the current testing fold
        for test_fold in fold_index:
            msg = '---------- Fold %d ' % test_fold
            log.info(msg + '-' * (80 - len(msg)))

            train_folds = tuple(i for i in fold_index if i != test_fold)
            # Train positive/negative IDs
            fold_train_positive_ids = set([
                uid
                for fidx in train_folds
                for uid in positive_ids[pos_fold_slices[fidx]]
            ])
            fold_train_negative_ids = set([
                uid
                for fidx in train_folds
                for uid in negative_ids[neg_fold_slices[fidx]]
            ])
            # Test positive/negative IDs
            fold_test_positive_ids = set(positive_ids[pos_fold_slices[test_fold]])
            fold_test_negative_ids = set(negative_ids[neg_fold_slices[test_fold]])

            # FeatureMemory objects for positive and negative folds
            fold_fm = FeatureMemory(fm.get_ids(), fold_train_negative_ids,
                                    fm.get_feature_matrix(),
                                    fm.get_kernel_matrix())
            fold_dk = fold_fm.get_distance_kernel()

            #
            # Training SVM model for current fold
            #
            # symmetric_submatrix call automatically includes the DK's BG set,
            # which was initialized above to be the fold's training negatives.
            idx2id_map, idx_bg_flags, m = \
                fold_dk.symmetric_submatrix(*fold_train_positive_ids)
            svm_train_labels = numpy.array(tuple(not b for b in idx_bg_flags))
            # Train the model with the current parameter set
            train_d = iqr_modules.iqr_model_train(m, svm_train_labels,
                                                  idx2id_map, param_set)
            fold_svm_model = train_d['model']
            fold_svm_svids = train_d['clipids_SVs']

            #
            # Model application to test fold
            #
            # Need testing kernel to include both testing positive and negative
            # IDs. Merging sets.
            fold_test_ids = set.union(fold_test_positive_ids,
                                      fold_test_negative_ids)
            idx2id_row, idx2id_col, kernel_test = \
                fold_dk.extract_rows(fold_svm_svids, col_ids=fold_test_ids)
            # adjust the contents of the testing kernel to only include the
            test_d = iqr_modules.iqr_model_test(fold_svm_model,
                                                kernel_test.A,
                                                idx2id_col)
            ordered_results = sorted(zip(test_d['clipids'], test_d['probs']),
                                     key=lambda x: x[1],
                                     reverse=True)

            # Store ordered scores and label lists for precision calculation
            # later.
            fold_results.append({
                'scores': [e[1] for e in ordered_results],
                'labels': [1 if (cid in fold_test_positive_ids) else 0
                           for cid, prob in ordered_results]
            })

        fold_precisions = perf_estimation.average_precision_R0(fold_results)

        # Average chosen precision metric across folds
        p_set_avg_precision.append(
            sum(fp[metric] for fp in fold_precisions) / k_folds
        )

    # Choose the best performing parameter set
    log.debug("Parameter precisions:\n%s",
              zip(parameter_sets, p_set_avg_precision))
    p_best_idx = p_set_avg_precision.index(max(p_set_avg_precision))
    p_best = parameter_sets[p_best_idx]
    log.info("Best chosen: %s", p_best)

    return p_best


if __name__ == '__main__':
    # MANUAL TESTING
    from masir.search.colordescriptor import ColorDescriptor
    import masir_config

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    # TODO: Add random search for base-line comparison

    pos_ids = list(numpy.loadtxt("positive_ids.txt", dtype=int))
    neg_ids = list(numpy.loadtxt("negative_ids.txt", dtype=int))

    cd_csift = ColorDescriptor.CSIFT(masir_config.DIR_DATA, masir_config.DIR_WORK)
    fm = FeatureMemory.construct_from_descriptor(cd_csift)
    parameter_set = [
        '-w1 50 -t 4 -b 1 -c 0.1',
        '-w1 50 -t 4 -b 1 -c 0.5',
        '-w1 50 -t 4 -b 1 -c 1',
        '-w1 50 -t 4 -b 1 -c 5',
        '-w1 50 -t 4 -b 1 -c 10',
        '-w1 50 -t 4 -b 1 -c 50',
        '-w1 50 -t 4 -b 1 -c 100',
        '-w1 50 -t 4 -b 1 -c 500',
    ]
    b = masir_svm_cross_validate(5, parameter_set,
                                 fm, pos_ids, neg_ids)

    print
    print "Best set:", b
    print
