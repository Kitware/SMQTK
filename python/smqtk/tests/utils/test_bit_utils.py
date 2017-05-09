import random
import unittest

import nose.tools
import numpy
# noinspection PyUnresolvedReferences
from six.moves import range

from smqtk.utils import bit_utils


class TestBitUtils (unittest.TestCase):

    def test_int_to_bit_vector_large_0(self):
        # Need at least one bit to represent 0.
        numpy.testing.assert_array_equal(
            bit_utils.int_to_bit_vector_large(0),
            [False]
        )
        # Force 5 bits.
        numpy.testing.assert_array_equal(
            bit_utils.int_to_bit_vector_large(0, 5),
            [False, False, False, False, False]
        )

    def test_int_to_bit_vector_large_1(self):
        numpy.testing.assert_array_equal(
            bit_utils.int_to_bit_vector_large(1),
            [True]
        )
        numpy.testing.assert_array_equal(
            bit_utils.int_to_bit_vector_large(1, 7),
            ([False] * 6) + [True]
        )

    def test_int_to_bit_vector_large_large(self):
        # Try large integer bit vectors
        int_val = (2**256) - 1
        expected_vector = [True] * 256
        numpy.testing.assert_array_equal(
            bit_utils.int_to_bit_vector_large(int_val),
            expected_vector
        )

        int_val = (2**512)
        expected_vector = [True] + ([False] * 512)
        numpy.testing.assert_array_equal(
            bit_utils.int_to_bit_vector_large(int_val),
            expected_vector
        )

    def test_int_to_bit_vector_large_invalid_bits(self):
        # Cannot represent 5 in binary using 1 bit.
        nose.tools.assert_raises(
            ValueError,
            bit_utils.int_to_bit_vector_large,
            5, 1
        )

    def test_popcount(self):
        nose.tools.assert_equal(bit_utils.popcount(1), 1)
        nose.tools.assert_equal(bit_utils.popcount(2), 1)
        nose.tools.assert_equal(bit_utils.popcount(3), 2)
        nose.tools.assert_equal(bit_utils.popcount(2 ** 16), 1)
        nose.tools.assert_equal(bit_utils.popcount(2**16 - 1), 16)
        nose.tools.assert_equal(bit_utils.popcount(2 ** 32), 1)
        nose.tools.assert_equal(bit_utils.popcount(2 ** 32 - 1), 32)

    def test_popcount_0(self):
        nose.tools.assert_equal(bit_utils.popcount(0), 0)

    def test_popcount_limits(self):
        # Make sure documented integer limit is truthful.
        c = 10000
        for _ in range(c):
            # noinspection PyUnresolvedReferences
            v = random.randint(0, bit_utils.popcount.v_max)
            # Known method to always work based on counting python's binary
            # string representation.
            v_bin_count = bin(v).count('1')
            # Test method
            v_pop_count = bit_utils.popcount(v)

            nose.tools.assert_equal(v_pop_count, v_bin_count,
                                    'popcount failed for integer %d' % v)
