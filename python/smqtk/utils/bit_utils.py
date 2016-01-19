import math

import numpy

from . import ncr

try:
    from numba import jit

except ImportError:

    def jit(func_or_sig):
        import types
        if isinstance(func_or_sig, (types.FunctionType, types.MethodType)):
            return func_or_sig
        else:
            return lambda *args, **kwds: func_or_sig


__author__ = "paul.tunison@kitware.com"


def next_perm(v):
    """
    Compute the lexicographically next bit permutation

    Generates next permutation with a given amount of set bits,
    given the previous lexicographical value.

    Taken from http://graphics.stanford.edu/~seander/bithacks.html

    """
    t = (v | (v - 1)) + 1
    w = t | ((((t & -t) / (v & -v)) >> 1) - 1)
    return w


def iter_perms(l, n):
    """
    Return an iterator over bit combinations of length ``l`` with ``n`` set
    bits.

    :raises StopIteration: If ``n`` <= 0 or normal completion.

    :param l: Total bit length to work with. The ``n`` in nCr problem.
    :type l: int

    :param n: Number of bits to be set in permutations. The ``r`` in nCr
        problem.
    :type n: int

    :return: List of bit vector permutations of the value ``(1<<n)-1`` over
        ``l`` bits.
    :rtype: list[int]

    """
    if n <= 0:
        raise StopIteration()
    n = min(l, n)
    s = (1 << n) - 1
    yield s
    for _ in xrange(ncr(l, n) - 1):
        s = next_perm(s)
        yield s


def neighbor_codes(b, c, d):
    """
    Iterate through integers of bit length ``b``, where ``b`` is the number
    of bits, that are ``d`` hamming distance away from query code ``c``.

    This will yield a number of elements equal to ``nCr(b, d)``.

    We expect ``d`` to be the integer hamming distance,
    e.g. h(001101, 100101) == 2, not 0.333.

    :param b: integer bit length
    :param b: int

    :param c: Query small-code integer
    :type c: int

    :param d: Integer hamming distance
    :type d: int

    """
    if not d:
        yield c
    else:
        for fltr in iter_perms(b, d):
            yield c ^ fltr


@jit
def bit_vector_to_int(v):
    """
    Transform a numpy vector representing a sequence of binary bits [0 | >0]
    into an integer representation.

    This version handles vectors of up to 64bits in size.

    :param v: 1D Vector of bits
    :type v: numpy.core.multiarray.ndarray

    :return: Integer equivalent
    :rtype: int

    """
    c = 0L
    for b in v:
        c = (c * 2L) + int(b)
    return c


def bit_vector_to_int_large(v):
    """
    Transform a numpy vector representing a sequence of binary bits [0 | >0]
    into an integer representation.

    This function is the special form that can handle very large integers
    (>64bit).

    :param v: 1D Vector of bits
    :type v: numpy.core.multiarray.ndarray

    :return: Integer equivalent
    :rtype: int

    """
    c = 0L
    for b in v:
        c = (c * 2L) + int(b)
    return c


@jit
def int_to_bit_vector(integer, bits=0):
    """
    Transform integer into a bit vector, optionally of a specific length.

    This version handles vectors of up to 64bits in size.

    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by the
        vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian), or None if too few
        ``bits`` were specified to contain the result.
    :rtype: numpy.ndarray[bool]

    """
    # Converting integer to array
    if integer:
        size = int(numpy.floor(numpy.log2(integer)) + 1)
    else:
        size = 0
    if bits and (bits - size) < 0:
        return None

    v = numpy.zeros(bits or size, numpy.bool_)
    for i in xrange(0, size):
        v[-(i+1)] = integer & 1
        integer >>= 1

    return v


def int_to_bit_vector_large(integer, bits=0):
    """
    Transform integer into a bit vector, optionally of a specific length.

    This function is the special form that can handle very large integers
    (>64bit).

    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by the
        vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian), or None if too few
        ``bits`` were specified to contain the result.
    :rtype: numpy.ndarray[bool]

    """
    # Converting integer to array'
    if integer:
        size = int(math.floor(math.log(integer, 2)) + 1)
    else:
        size = 0
    if bits and (bits - size) < 0:
        return None

    v = numpy.zeros(bits or size, numpy.bool_)
    for i in xrange(0, size):
        v[-(i+1)] = integer & 1
        integer >>= 1

    return v
