from __future__ import division, print_function
import random
import unittest

import nose.tools

from smqtk.utils.parallel import parallel_map
from six.moves import range


class TestParallelMap (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 10000

        # Random characters in range [a, z]
        cls.test_string = [chr(random.randint(97, 122)) for _ in range(n)]
        cls.test_func = ord
        # Since this parallel function is intended to perform similar to the
        # built-in map function.
        cls.expected = list(map(cls.test_func, cls.test_string))

    def test_simple_ordered_threaded(self):
        # Make sure results are still in order as requested
        r = list(parallel_map(self.test_func, self.test_string,
                              ordered=True, use_multiprocessing=False))
        nose.tools.assert_equal(r, self.expected)

    def test_simple_ordered_multiprocess(self):
        r = list(parallel_map(self.test_func, self.test_string,
                              ordered=True, use_multiprocessing=True))
        nose.tools.assert_equal(r, self.expected)

    def test_simple_unordered_threaded(self):
        r = list(parallel_map(self.test_func, self.test_string,
                              ordered=False, use_multiprocessing=False))
        nose.tools.assert_equal(set(r), set(self.expected))

    def test_simple_unordered_multiprocess(self):
        r = list(parallel_map(self.test_func, self.test_string,
                              ordered=False, use_multiprocessing=True))
        nose.tools.assert_equal(set(r), set(self.expected))

    def test_exception_handing_threaded(self):
        def raise_ex(_):
            raise RuntimeError("Expected exception")

        nose.tools.assert_raises(
            RuntimeError,
            list,
            parallel_map(raise_ex, [1], use_multiprocessing=False)
        )

    def test_exception_handing_multiprocess(self):
        def raise_ex(_):
            raise RuntimeError("Expected exception")

        nose.tools.assert_raises(
            RuntimeError,
            list,
            parallel_map(raise_ex, [1], use_multiprocessing=True)
        )

    def test_multisequence(self):
        def test_func(a, b, c):
            return a + b + c

        s1 = [1] * 10
        s2 = [2] * 10
        s3 = [3] * 10
        r = list(parallel_map(test_func, s1, s2, s3,
                              use_multiprocessing=False))

        expected = [6] * 10
        nose.tools.assert_equal(r, expected)

    def test_multisequence_short_cutoff(self):
        def test_func(a, b, c):
            return a + b + c

        s1 = [1] * 10
        s2 = [2] * 4
        s3 = [3] * 10
        r = list(parallel_map(test_func, s1, s2, s3,
                              use_multiprocessing=False,
                              ordered=True))

        exp = [6] * 4
        nose.tools.assert_equal(r, exp)

    def test_multisequence_fill_void(self):
        def test_func(a, b, c):
            return a + b + c

        s1 = [1] * 10
        s2 = [2] * 4
        s3 = [3] * 10
        r = list(parallel_map(test_func, s1, s2, s3,
                              use_multiprocessing=False,
                              fill_void=10,
                              ordered=True))

        expected = [6] * 4 + [14] * 6
        nose.tools.assert_equal(r, expected)

    def test_nested_multiprocessing(self):
        # char -> char -> ord -> char -> ord
        g1 = parallel_map(lambda e: e, self.test_string,
                          ordered=True,
                          use_multiprocessing=True,
                          cores=2)
        g2 = parallel_map(ord, g1,
                          ordered=True,
                          use_multiprocessing=True,
                          cores=2)
        g3 = parallel_map(chr, g2,
                          ordered=True,
                          use_multiprocessing=True,
                          cores=2)
        g4 = parallel_map(ord, g3,
                          ordered=True,
                          use_multiprocessing=True,
                          cores=2)

        expected = list(map(ord, self.test_string))
        nose.tools.assert_equal(
            list(g4),
            expected
        )

    def test_nested_threading(self):
        # char -> char -> ord -> char -> ord
        g1 = parallel_map(lambda e: e, self.test_string,
                          ordered=True,
                          use_multiprocessing=False,
                          cores=2,
                          name='g1')
        g2 = parallel_map(ord, g1,
                          ordered=True,
                          use_multiprocessing=False,
                          cores=2,
                          name='g2')
        g3 = parallel_map(chr, g2,
                          ordered=True,
                          use_multiprocessing=False,
                          cores=2,
                          name='g3')
        g4 = parallel_map(ord, g3,
                          ordered=True,
                          use_multiprocessing=False,
                          cores=2,
                          name='g4')

        expected = list(map(ord, self.test_string))
        nose.tools.assert_equal(
            list(g4),
            expected
        )
