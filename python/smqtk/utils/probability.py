from __future__ import (absolute_import, division, print_function)

import numpy as np


def adjust_proba(proba, adj):
    """
    Adjust the probabilities in `proba` with the log-scale adjustments `adj`.
    If the probabilities in a row of `proba` are `(p1,...,pd)`, and the
    adjustments are `(a1,...,ad)`, then the output will be:

        `(p1*exp(a1),...,pd*exp(ad)) / sum(p1*exp(a1),...,pd*exp(ad))`.

    The output is therefore scale-invariant in `proba`, so `proba` need not
    sum to one, but must contain all nonnegative values and at least one
    element per row must be positive.

    This can be used to modify predictions from (e.g.) classifiers, such as
    when the training sizes are imbalanced. Applying
    `(-log(card1),...,-log(cardd)`, where `cardi` is the cardinality of the
    class in the corresponding training set, would lower the recall for high
    cardinality classes and boost the recall for low cardinality classes.[1]

    An alternative interpretation of this function is that if a row of `proba`
    is a "prior," and `adj` is the logarithm of the "evidence," then the
    corresponding output row is the "posterior" after applying a Bayesian
    update. In the example above, our evidence is that the classes are skewed
    so we want to adjust the output to compensate.

    The results are *additive in `adj`* for multiple calls of the function.
    That is, `adjust_proba(adjust_proba(proba, adj1), adj2)` is simply
    `adjust_proba(proba, adj1+adj2)`. This means that subsequent calls are
    unnecessary and adjustments may be accumulated and applied all at once.
    (This follows from the fact that successive Bayesian updates are
    equivalent to updating against the product of the evidence.)

    It is also the case that if `adj` is a constant vector, then
    `np.allclose(adjust_proba(proba, adj), proba)` will return true. (This
    follows from the fact that uniform evidence has no effect in Bayesian
    updates.) Combined with the previous fact, this means that `adj` may take
    values of either (1,0,0) or (0,-1,-1), for example, and produce the same
    output for the same `proba` (since `(1,0,0) = (1,1,1) + (0,-1,-1)`).

    :param proba: Numpy array of shape (n,d) with probabilities in each row.
       Each column represents a class. The probabilities need not be
       normalized to sum to 1.
    :type proba: np.ndarray | collections.Iterable

    :param adj: An iterable with d elements or a numpy array of shape (d,)
        that contains the adjustments for each class. The adjustments are
        log-scale and the probabilities in `proba` will be multiplied by the
        exponential of `adj`, then normalized. The parameters in `adj` need
        not be normalized.
    :type adj: np.ndarray | collections.Iterable

    :return: A numpy array with the same shape as `proba` containing the
        adjusted probabilities per-class. The output will sum to one on the
        second axis.
    :rtype: np.ndarray

    [1] Note: If `proba` is a set of probabilistic predictions of a calibrated
    classifier, it is likely that applying adjustments to the predictions will
    affect accuracy negatively.
    """
    adj_proba, adj_vals = np.atleast_2d(proba, adj)

    if adj_proba.shape[1] != adj_vals.shape[1]:
        raise ValueError(
            "The dimensions of probabilities and adjustments must be "
            "compatible (number of columns must be equal).")

    if np.any(adj_proba < 0):
        raise ValueError("Probabilities must be at least 0.")
    # After eliminating the possibility of a negative value, we only need to
    # test if the sum is close to zero
    if np.isclose(adj_proba.sum(), 0):
        raise ValueError("At least one probability must be positive.")

    # Prevent exponential overflow (this cancels out on return)
    adj_proba = adj_proba * np.exp(adj_vals - adj_vals.max())

    # Normalize and return adjusted probabilities
    return adj_proba / adj_proba.sum(axis=1, keepdims=True)
