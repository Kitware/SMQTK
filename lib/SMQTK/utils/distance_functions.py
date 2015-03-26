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
    vectors i and j, returning a value between 0.0 and 1.0. 1.0 means full
    intersection, and 0.0 means no intersection.

    This implements non-branching formula for efficient computation.

    :param i: Histogram i
    :type i: numpy.core.multiarray.ndarray

    :param j: Histogram j
    :type j: numpy.core.multiarray.ndarray

    :return: Float percent intersection amount.
    :rtype: float

    """
    # TODO: Always normalize input histograms here? e.g.:
    #       ...
    #       i_sum = i.sum()
    #       if i_sum != 0:
    #           i /= i_sum
    #       ...
    return (i + j - numpy.abs(i - j)).sum() * 0.5
