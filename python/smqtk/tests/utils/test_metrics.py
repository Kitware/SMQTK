from __future__ import division, print_function
import random
import unittest

import nose.tools as ntools
import numpy as np

from six.moves import range
from smqtk.utils import metrics as df


__author__ = "paul.tunison@kitware.com"


def gen(n):
    return random.randint(0, 2**n-1)


class TestHistogramIntersectionDistance (unittest.TestCase):

    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0, 1])
    v4 = np.array([.5, .5])

    m1 = np.array([v2, v3, v4])
    m2 = np.array([v2, v4])

    hi_methods = [
        df.histogram_intersection_distance_fast,
        df.histogram_intersection_distance,
    ]

    def test_hi_result_zerovector(self):
        # HI distance of anything with the zero vector is 1.0 since a valid
        # histogram does not intersect with nothing.
        # REMEMBER we're talking about distance here, not similarity
        for m in self.hi_methods:
            print("Tests for method: %s" % m)
            ntools.assert_equal(m(self.v1, self.v1), 1.)

            ntools.assert_equal(m(self.v1, self.v2), 1.)
            ntools.assert_equal(m(self.v2, self.v1), 1.)

            ntools.assert_equal(m(self.v1, self.v4), 1.)
            ntools.assert_equal(m(self.v4, self.v1), 1.)

    def test_hi_result_normal(self):
        for m in self.hi_methods:
            print("Tests for method: %s" % m)

            ntools.assert_equal(m(self.v2, self.v3), 1.)
            ntools.assert_equal(m(self.v3, self.v2), 1.)

            ntools.assert_equal(m(self.v2, self.v4), 0.5)
            ntools.assert_equal(m(self.v4, self.v2), 0.5)

            ntools.assert_equal(m(self.v3, self.v4), 0.5)
            ntools.assert_equal(m(self.v4, self.v3), 0.5)

            ntools.assert_equal(m(self.v4, self.v4), 0.0)

    def test_hi_input_format(self):
        # the general form method should be able to take any combination of
        # vectors and matrices, following documented rules.

        ntools.assert_equal(
            df.histogram_intersection_distance(self.v4, self.v3),
            0.5
        )

        np.testing.assert_array_equal(
            df.histogram_intersection_distance(self.v2, self.m1),
            [0., 1., 0.5]
        )
        np.testing.assert_array_equal(
            df.histogram_intersection_distance(self.m1, self.v2),
            [0., 1., 0.5]
        )

        np.testing.assert_array_equal(
            df.histogram_intersection_distance(self.m1, self.m1),
            [0, 0, 0]
        )

        ntools.assert_raises(ValueError, df.histogram_intersection_distance,
                             self.m1, self.m2)


class TestHammingDistance (unittest.TestCase):

    def test_hd_0(self):
        ntools.assert_equal(df.hamming_distance(0, 0), 0)

    def test_rand(self):
        n = 64
        for i in range(1000):
            a = gen(n)
            b = gen(n)
            actual = bin(a^b).count('1')
            ntools.assert_equal(df.hamming_distance(a, b), actual)

    def test_rand_large(self):
        n = 1024
        for i in range(1000):
            a = gen(n)
            b = gen(n)
            actual = bin(a^b).count('1')
            ntools.assert_equal(df.hamming_distance(a, b), actual)
