import random
import unittest

from smqtk.utils import factors


# First 100 known primes
p100 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
        139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
        211,  223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
        281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
        367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
        443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521,
        523, 541]


class TestSoe (unittest.TestCase):
    """
    Unit tests for caching sieve of eratosthenes function.
    """

    def setUp(self):
        """
        Clear SoE caches before each test.
        """
        # Values copied from head of factors.py file, simulating initial import.
        factors._soe_prime_cache = [2, 3]
        factors._soe_not_prime_map = {9: 3}
        factors._soe_c = 5

    def test_0(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(0),
            []
        )

    def test_1(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(1),
            []
        )

    def test_2(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(2),
            [2]
        )

    def test_3(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(3),
            [2, 3]
        )

    def test_4(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(4),
            [2, 3]
        )

    def test_5(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(5),
            [2, 3, 5]
        )

    def test_7(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(7),
            [2, 3, 5, 7]
        )

    def test_15(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(15),
            [2, 3, 5, 7, 11, 13]
        )

    def test_25(self):
        self.assertListEqual(
            factors.sieve_of_eratosthenes(25),
            [2, 3, 5, 7, 11, 13, 17, 19, 23]
        )

    def test_100_primes(self):
        self.assertListEqual(factors.sieve_of_eratosthenes(541), p100)

    def test_100_with_caching(self):
        # First ask for a value with 99 prime returns, then a value with 100 to
        # see that we did get that new extra value.
        self.assertListEqual(factors.sieve_of_eratosthenes(540), p100[:-1])
        self.assertListEqual(factors.sieve_of_eratosthenes(541), p100)

    def test_100_random_with_reset(self):
        # Randomly test for prime values <= than 100th prime, resetting the
        # cache each time.
        for _ in range(100):
            r = random.randint(0, max(p100)+1)
            expected_primes = [p for p in p100 if p <= r]
            self.setUp()  # Reset
            actual_primes = factors.sieve_of_eratosthenes(r)
            self.assertListEqual(expected_primes, actual_primes,
                                 "Unexpected return for query value: %d" % r)

    def test_100_random_with_caching(self):
        # Randomly test for prime values <= than 100th prime, NOT resetting the
        # cache each time.
        for _ in range(100):
            r = random.randint(0, max(p100) + 1)
            expected_primes = [p for p in p100 if p <= r]
            actual_primes = factors.sieve_of_eratosthenes(r)
            self.assertListEqual(expected_primes, actual_primes,
                                 "Unexpected return for query value: %d" % r)
