from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import unittest

import numpy as np

from smqtk.utils import prob_utils


class TestAdjustProba (unittest.TestCase):

    def test_single_class(self):

        num = 10
        dim = 1

        proba = np.random.rand(num, dim)
        self.assertTrue(np.allclose(1, prob_utils.adjust_proba(proba, [1])))

        proba = np.random.rand(num, dim)
        self.assertTrue(np.allclose(1, prob_utils.adjust_proba(proba, [-1])))

        proba = np.ones_like(proba)
        self.assertTrue(np.allclose(1, prob_utils.adjust_proba(proba, [1])))

    def test_failure_cases(self):

        num = 10
        dim = 3

        proba = np.zeros((num, dim))
        self.assertRaisesRegexp(ValueError,
                                "At least one probability must be positive",
                                prob_utils.adjust_proba, proba, [1, 2, 3])

        proba[1] = -1.0
        proba[2] = 1.0
        self.assertRaisesRegexp(ValueError,
                                "Probabilities must be at least 0",
                                prob_utils.adjust_proba, proba, [1, 2, 3])

    def test_shape_cases(self):

        num = 10
        dim = 3

        proba = np.random.rand(num, dim)
        self.assertRaisesRegexp(ValueError,
                                "The dimensions of probabilities and "
                                "adjustments must be compatible.",
                                prob_utils.adjust_proba, proba, [1, 2])

        proba = np.random.rand(1, dim)
        proba /= proba.sum()
        self.assertTrue(
            np.allclose(proba, prob_utils.adjust_proba(proba, [1, 1, 1])))

        self.assertRaisesRegexp(ValueError,
                                "The dimensions of probabilities and "
                                "adjustments must be compatible.",
                                prob_utils.adjust_proba,
                                np.ones((num, 1)), np.ones((1, num)))

    def test_adjust_constant(self):

        num = 10
        dim = 3

        proba = np.random.rand(num, dim)
        proba /= proba.sum(axis=1, keepdims=True)

        self.assertTrue(
            np.allclose(proba, prob_utils.adjust_proba(proba, [1, 1, 1])))
        self.assertTrue(
            np.allclose(proba, prob_utils.adjust_proba(proba, [10, 10, 10])))

    def test_adjust_serial_vs_sum(self):

        num = 10
        dim = 3

        proba = np.random.rand(num, dim)
        proba /= proba.sum(axis=1, keepdims=True)

        adj1 = np.array([1, 2, 3])
        adj2 = np.array([2, 0, -2])

        proba_fst = prob_utils.adjust_proba(proba, adj1)
        proba_snd = prob_utils.adjust_proba(proba_fst, adj2)
        proba_sum = prob_utils.adjust_proba(proba, adj1 + adj2)

        self.assertTrue(np.allclose(proba_snd, proba_sum))

        proba_fst = prob_utils.adjust_proba(proba, adj1)
        proba_snd = prob_utils.adjust_proba(proba_fst, -adj1)

        self.assertTrue(np.allclose(proba_snd, proba))

    def test_adjust(self):

        num = 10
        dim = 3

        proba = np.random.rand(num, dim)
        proba /= proba.sum(axis=1, keepdims=True)

        adj = [0, 1, 0]

        proba_post = prob_utils.adjust_proba(proba, adj)
        comp = proba_post > proba
        self.assertTrue(np.all([False, True, False] == comp))
        comp = proba_post < proba
        self.assertTrue(np.all([True, False, True] == comp))
        comp = np.isclose(proba, proba_post)
        self.assertFalse(np.any(comp))

        adj = [-1, 0, 0]

        proba_post = prob_utils.adjust_proba(proba, adj)
        comp = proba_post < proba
        self.assertTrue(np.all([True, False, False] == comp))
        comp = proba_post > proba
        self.assertTrue(np.all([False, True, True] == comp))
        comp = np.isclose(proba, proba_post)
        self.assertFalse(np.any(comp))

        adj = [1.5, 0, -1.5]

        proba_post = prob_utils.adjust_proba(proba, adj)
        comp = proba_post < proba
        self.assertTrue(np.all([False, True] == comp[:, [0, 2]]))
        comp = proba_post > proba
        self.assertTrue(np.all([True, False] == comp[:, [0, 2]]))
        comp = np.isclose(proba, proba_post)
        self.assertFalse(np.all([False, True, False] == comp))
