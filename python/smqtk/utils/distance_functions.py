"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import gmpy2
import numpy as np


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
    return 1.0 - ((i + j - np.abs(i - j)).sum() * 0.5)


def histogram_intersection_distance2(a, b):
    """
    Compute the histogram intersection distance between given histogram
    vectors or matrices ``a`` and ``b``, returning a value between ``0.0`` and
    ``1.0``. This is the inverse of intersection similarity, whereby a distance
    of  ``0.0`` means full intersection and ``1.0`` means no intersection.

    This implements non-branching formula for efficient computation.

    Input vectors ``a`` and ``b`` may be of 1 or 2 dimensions. Depending on the
    values of ``a`` and ``b``, different things may occur:

    * If both ``a`` and ``b`` are 2D matrices, they're shape must be congruent
      and the result will be an vector of distances between parallel rows.

    * If either is a 1D vector and the other is a 2D matrix, a vector of
      distances is returned between the 1D vector and each row of the 2D matrix.

    * If both ``a`` and ``b`` are 1D vectors, a floating-point scalar distance
      is returned that is the histogram intersection distance between the input
      vectors.

    :param a: Histogram or array of histograms ``a``
    :type a: numpy.core.multiarray.ndarray

    :param b: Histogram or array of histograms ``b``
    :type b: numpy.core.multiarray.ndarray

    :return: Float or array of float distance (inverse percent intersection).
    :rtype: float or numpy.core.multiarray.ndarray
    """
    # TODO: input value checks?
    # Which axis to sum on. If there is a matrix involved, its along column,
    # but if its just two arrays its along the row.
    # The following is noticeably slower:
    #   sum_axis = not (a.ndim == 1 and b.ndim == 1)
    sum_axis = 1
    if a.ndim == 1 and b.ndim == 1:
        sum_axis = 0
    return 1. - ((np.add(a, b) - np.abs(np.subtract(a, b))).sum(sum_axis) * 0.5)


def euclidean_distance(i, j):
    """
    Compute euclidian distance between two N-dimensional point vectors.

    :param i: Vector i
    :type i: numpy.core.multiarray.ndarray

    :param j: Vector j
    :type j: numpy.core.multiarray.ndarray

    :return: Float distance.
    :rtype: float

    """
    # noinspection PyTypeChecker
    return np.sqrt(np.power(i - j, 2.0).sum())


def cosine_similarity(i, j):
    """
    Angular similarity between vectors i and j. Results in a value between 1,
    where i and j are exactly the same, to -1, meaning exactly opposite. 0
    indicates orthogonality. Negative values will only be returned if input
    vectors can have negative values.

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
    return np.dot(i, j) / (np.sqrt(i.dot(i)) * np.sqrt(j.dot(j)))


def hamming_distance(i, j):
    """
    Return the hamming distance between the two given integers, or the number of
    places where the bits differ.

    :param i: First integer.
    :type i: int
    :param j: Second integer.
    :type j: int

    :return: Integer hamming distance between the two values.
    :rtype: int

    """
    return gmpy2.popcount(i ^ j)
