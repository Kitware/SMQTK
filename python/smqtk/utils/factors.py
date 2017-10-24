# -*- coding: utf-8 -*-

import math


__all__ = ["sieve_of_eratosthenes", "prime_factors", "factors", "factor_pairs"]


# Set of known prime values
_soe_prime_cache = [2, 3]
# Map of not-prime values and the increment after that value where the next
# not-prime value may be found. This should only contain keys greater-than
# the max(_prime_cache) value for finding further prime values.
_soe_not_prime_map = {4: 2, 9: 3}
# The next value to continue iterating from.
_soe_c = 4
# TODO: Lock these globals when in use.


def sieve_of_eratosthenes(N):
    """
    Return an ascending ordered list of prime numbers up to the given value,
    generated via the sieve of eratosthenes method.

    TODO: Add starting offset so as to be able to get primes within a range.

    :param N: Value to get prime values less-than-or-equal this value.
    :return: List of prime integer values <= N.

    """
    # TODO: Optimization - all primes after 2 are odd, so special case when
    #           N == 2, else loop over and deal only with odd numbers.
    global _soe_c

    # in case we get N as a string or something
    N = float(N)

    if _soe_c > N:
        # We have already computed all primes up to N, so return cache-filled
        # list.
        return [p for p in _soe_prime_cache if p <= N]

    # We need to find more prime values, so start with the least known
    # not-prime and sieve our way up.
    while _soe_c <= N:
        if _soe_c in _soe_not_prime_map:
            # Get increment for the current non-prime value and record the next
            # non-prime value, keeping this increment. Incrementing past N in
            # order to main state in reused cache structure.
            incr = _soe_not_prime_map[_soe_c]
            nnp = _soe_c + incr
            while (nnp in _soe_not_prime_map):
                nnp += incr
            _soe_not_prime_map[nnp] = incr
            # Remove _soe_c from map as it has been considered now.
            del _soe_not_prime_map[_soe_c]
        else:
            # _soe_c is a prime value: record it and add square of _soe_c to
            # not-prime-map (square of the prime is the first value not
            # attaiable by multiplying preceeding prime value).
            _soe_prime_cache.append(_soe_c)
            _soe_not_prime_map[_soe_c*_soe_c] = _soe_c
        _soe_c += 1

    # since we had to extend it, return a copy of the cache
    return list( _soe_prime_cache )


def prime_factors(N):
    """
    Returns ordered ascending prime factors of N.
    """
    sqrt_N = math.sqrt(N)
    primes = sieve_of_eratosthenes(sqrt_N)

    # prime factorize N
    p_factors = []
    remaining = N
    for p in primes:
        # Short-cut loop exit when we know we cant factorize any more
        if remaining == 1:
            break
        # factory out
        while remaining > 1 and ((remaining % p) == 0):
            p_factors.append(p)
            remaining /= p

    if remaining != 1:
        p_factors.append(remaining)

    return p_factors


def factors(N):
    """
    Return divisors/factors of N.
    """
    pf = prime_factors(N)
    # noinspection PySetFunctionToLiteral
    ftors = set([1])

    # Compute product of all primes
    for p in pf:
        new_factors = set()
        # multiply the current prime factor with all current factors in order to
        # iteratively build up component factors of N
        for e in ftors:
            new_factors.add(e * p)
        ftors.update(new_factors)

    # using sorted as we want to return a list, and directly converting a set
    # to a list causes things to be out of sequential order.
    return ftors


def factor_pairs(N):
    """
    Return a list of factor pairs of N, whose individual product produce N.
    """
    ftors = factors(N)
    # since what comes out of factors is sorted, step indices from the edges
    # towards the center, pairing the edges to produce factor pairs.
    i = 0
    j = len(ftors) - 1
    pairs = []
    while i <= j:
        pairs.append((ftors[i], ftors[j]))
        i += 1
        j -= 1
    return pairs
