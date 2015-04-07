"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy


def histogram_intersection_distance(i, j):
    """
    Compute the histogram intersection percent relation between given histogram
    vectors ``a`` and ``b``, returning a value between 0.0 and 1.0. 0.0 means
    full intersection, and 1.0 means no intersection.

    This implements non-branching formula for efficient computation.

    :param i: Histogram ``i``
    :type i: numpy.core.multiarray.ndarray

    :param j: Histogram ``j``
    :type j: numpy.core.multiarray.ndarray

    :return: Float inverse percent intersection amount.
    :rtype: float

    """
    # TODO: Always normalize input histograms here? e.g.:
    #       ...
    #       i_sum = i.sum()
    #       if i_sum != 0:
    #           i /= i_sum
    #       ...
    return 1.0 - ((i + j - numpy.abs(i - j)).sum() * 0.5)


def histogram_intersection_dist_matrix(a, b):
    """
    Compute the histogram intersection percent relation between given histogram
    vectors ``a`` and ``b``, returning a value between 0.0 and 1.0. 0.0 means
    full intersection, and 1.0 means no intersection.

    This implements non-branching formula for efficient computation.

    Input vectors ``a`` and ``b`` may be of 1 or 2 dimensions. ``b`` must have
    dimensionality less than or equal to ``a``. The following chart shows what
    this function does depending on input format:

    * ``i.ndim == 1 && j.ndim == 1 && a.shape == b.shape``
        The distance between vectors ``a`` and ``b`` are returned as a floating
        point value

    * ``i.ndim == 2 && j.ndim == 1 && a.shape[0] == b.shape``
        A vector of distances are returned that correspond to ``d(a_i, j)``
        where ``i`` is the i\ :sup:`th` row of of ``a``.

    * ``i.ndim == 2 && j.ndim == 2 && a.shape == b.shape``
        A vector of distances are returned that correspond to ``d(a_i, b_i)``
        where ``i`` is the i\ :sup:`th` rows of ``a`` and ``b``.

    :raises ValueError: Input array shapes are incompatible.

    :param a: Histogram or array of histograms ``a``
    :type a: numpy.core.multiarray.ndarray

    :param b: Histogram or array of histograms ``b``
    :type b: numpy.core.multiarray.ndarray

    :return: Float or array of float distance (inverse percent intersection).
    :rtype: float or numpy.core.multiarray.ndarray
    """
    # TODO: Always normalize input histograms here? e.g.:
    #       ...
    #       i_sum = i.sum()
    #       if i_sum != 0:
    #           i /= i_sum
    #       ...
    a_dim = a.ndim
    if a_dim == 1:
        a = numpy.array([a])
    # TODO: Assert shape consistency

    r = 1.0 - ((numpy.add(a, b) - numpy.abs(numpy.subtract(a, b))).sum(axis=1) * 0.5)

    if a_dim == 1:
        return r[0]
    else:
        return r


def euclidian_distance(i, j):
    """
    Compute euclidian distance between two N-dimensional point vectors.

    :param i: Vector i
    :type i: numpy.core.multiarray.ndarray

    :param j: Vector j
    :type j: numpy.core.multiarray.ndarray

    :return: Float distance.
    :rtype: float

    """
    return numpy.sqrt(numpy.power(i - j, 2.0).sum())


def cosine_similarity(i, j):
    """
    Angular similarity between vectors i and j. Results in a value between 1,
    here i and j are exactly the same, to -1, meaning exactly opposite. 0
    indicates orthogonality.

    See: http://en.wikipedia.org/wiki/Cosine_similarity

    :param i: Vector i
    :type i: numpy.core.multiarray.ndarray

    :param j: Vector j
    :type j: numpy.core.multiarray.ndarray

    :return: Float similarity.
    :rtype: float

    """
    # numpy.linalg.norm is Frobenius norm (vector magnitude)
    # return numpy.dot(i, j) / (numpy.linalg.norm(i) * numpy.linalg.norm(j))

    # speed optimization, numpy.linalg.norm can be a bottleneck
    return numpy.dot(i, j) / (numpy.sqrt(i.dot(i)) * numpy.sqrt(j.dot(j)))
