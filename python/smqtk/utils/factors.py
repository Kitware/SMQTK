# -*- coding: utf-8 -*-

import math


__all__ = ["sieve_of_eratosthenes", "prime_factors", "all_factors"]


def sieve_of_eratosthenes(N):
    """
    Return an ascending ordered list of prime numbers up to the given value,
    generated via the sieve of eratosthenes method.

    TODO: Add starting offset so as to be able to get primes within a range
            / add dynamic programming concepts.
    """
    # TODO: Optimization - all primes after 2 are odd, so special case when
    #           N == 2, else loop over and deal only with odd numbers.

    # in case we get N as a string or something
    N = float(N)

    # maps non-prime number to an increment value defining the next value after
    # the key that is not a prime.
    not_primes = {}
    primes = []

    c = 2
    while c <= N:
        if c in not_primes:
            i = not_primes[c]
            # next non-prime number. Need to skip entries already in not_primes
            # so we don't interfere with recorded key-interval pairings
            nnp = c + i
            while (nnp in not_primes) and (nnp < N):
                nnp += i
            # npp may become >N, meaning results of this interval have already
            # been fully covered, and we can safely assign it out of the range
            # of N as we know we won't be looking past N at the most in the map.
            not_primes[nnp] = i
            # Won't be looking at c again, so we can remove it from the map to
            # save memory
            del not_primes[c]
        else:
            primes.append(c)
            # The lowest value that will
            c_squared = c*c
            if c_squared <= N:
                not_primes[c_squared] = c
        c += 1

    return primes


def prime_factors(N):
    """
    Returns ordered ascending prime factors of N.

    TODO: Could add caching of answers for quicker successive responses.
          Could make this recursive and leverage dynamic programming concepts
            -> find a prime factor, call prime_factors on remaining divisor
    """
    sqrt_N = math.sqrt(N)
    # sqrt_N = math.sqrt(N) + math.log10(N)
    primes = sieve_of_eratosthenes(sqrt_N)

    # prime factorize N
    p_factors = []
    remaining = N
    for p in primes:
        # Short-cut loop exit when we know we cant factorize any more
        if remaining == 1:
            break
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
    return sorted(ftors)


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