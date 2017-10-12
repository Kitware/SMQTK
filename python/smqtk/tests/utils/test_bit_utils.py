from __future__ import print_function

import numpy
import unittest

from six.moves import range

from smqtk.utils.bit_utils import (
    bit_vector_to_int,
    int_to_bit_vector,
    bit_vector_to_int_large,
    int_to_bit_vector_large,
)


class TestLargeBitVecConversions (unittest.TestCase):

    def setUp(self):
        one_bits = (254, 253, 2, 1, 0)
        standard_bits = [int(i in one_bits) for i in range(256)]
        self.boolvec = numpy.array(standard_bits, dtype='bool')
        self.intvec = numpy.array(standard_bits, dtype='int')
        self.int_rep = sum(2L**(255 - x) for x in one_bits)
        self.rev_int_rep = sum(2L**x for x in one_bits)

    def test_bit_vector_to_int_large(self):
        self.assertEqual(self.int_rep, bit_vector_to_int_large(self.boolvec))
        self.assertEqual(self.int_rep, bit_vector_to_int_large(self.intvec))
        self.assertEqual(
            self.rev_int_rep, bit_vector_to_int_large(self.boolvec[::-1])
        )
        self.assertEqual(
            self.rev_int_rep, bit_vector_to_int_large(self.intvec[::-1])
        )

    def test_int_to_bit_vector_large(self):
        numpy.testing.assert_array_equal(
            self.boolvec, int_to_bit_vector_large(self.int_rep, bits=256)
        )
        numpy.testing.assert_array_equal(
            self.boolvec[::-1],
            int_to_bit_vector_large(self.rev_int_rep, bits=256)
        )


class TestBitVecConversions (unittest.TestCase):

    def setUp(self):
        one_bits = (62, 61, 2, 1, 0)
        standard_bits = [int(i in one_bits) for i in range(64)]
        self.boolvec = numpy.array(standard_bits, dtype='bool')
        self.intvec = numpy.array(standard_bits, dtype='int')
        self.int_rep = sum(2L**(63 - x) for x in one_bits)
        self.rev_int_rep = sum(2L**x for x in one_bits)

    def test_bit_vector_to_int(self):
        self.assertEqual(self.int_rep, bit_vector_to_int(self.boolvec))
        self.assertEqual(self.int_rep, bit_vector_to_int(self.intvec))
        self.assertEqual(
            self.rev_int_rep, bit_vector_to_int(self.boolvec[::-1])
        )
        self.assertEqual(
            self.rev_int_rep, bit_vector_to_int(self.intvec[::-1])
        )

    def test_int_to_bit_vector(self):
        numpy.testing.assert_array_equal(
            self.boolvec, int_to_bit_vector(self.int_rep)
        )
        numpy.testing.assert_array_equal(
            self.boolvec[::-1],
            int_to_bit_vector(self.rev_int_rep, bits=64)
        )
