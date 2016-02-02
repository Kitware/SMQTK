import random
import unittest

import nose.tools

from smqtk.utils.parallel import parallel_map


class TestParallelMap (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 10000

        # Random characters in range [a, z]
        cls.test_string = [chr(random.randint(97, 122)) for _ in xrange(n)]
        cls.test_func = ord
        cls.expected = map(cls.test_func, cls.test_string)

    def test_simple_ordered_threaded(self):
        # Make sure results are still in order as requested
        r = list(parallel_map(self.test_string, self.test_func,
                              ordered=True, use_multiprocessing=False))
        nose.tools.assert_equal(r, self.expected)

    def test_simple_ordered_multiprocess(self):
        r = list(parallel_map(self.test_string, self.test_func,
                              ordered=True, use_multiprocessing=True))
        nose.tools.assert_equal(r, self.expected)

    def test_simple_unordered_threaded(self):
        r = list(parallel_map(self.test_string, self.test_func,
                              ordered=False, use_multiprocessing=False))
        nose.tools.assert_equal(set(r), set(self.expected))

    def test_simple_unordered_multiprocess(self):
        r = list(parallel_map(self.test_string, self.test_func,
                              ordered=False, use_multiprocessing=True))
        nose.tools.assert_equal(set(r), set(self.expected))

    def test_exception_handing_threaded(self):
        def raise_ex(_):
            raise RuntimeError("Expected exception")

        nose.tools.assert_raises(
            RuntimeError,
            list,
            parallel_map([1], raise_ex, use_multiprocessing=False)
        )

    def test_exception_handing_multiprocess(self):
        def raise_ex(_):
            raise RuntimeError("Expected exception")

        nose.tools.assert_raises(
            RuntimeError,
            list,
            parallel_map([1], raise_ex, use_multiprocessing=True)
        )
