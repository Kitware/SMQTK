import math

import numpy

# noinspection PyUnresolvedReferences
from six.moves import range

from . import ncr

try:
    from numba import jit
except (ImportError, TypeError):
    # Create passthrough function if numba is not installed.
    def jit(func_or_sig):
        import types
        if isinstance(func_or_sig, (types.FunctionType, types.MethodType)):
            return func_or_sig
        else:
            return lambda *args, **kwds: func_or_sig


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
    for _ in range(ncr(l, n) - 1):
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

    :raises ValueError: If ``bits`` specified is smaller than the required bits
        to represent the given ``integer`` value.

    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by the
        vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian).
    :rtype: numpy.ndarray[bool]

    """
    # Can't use math version because floating-point precision runs out after
    # about 2^48
    # -2 to remove length of '0b' string prefix
    size = len(bin(integer)) - 2

    if bits and (bits - size) < 0:
        raise ValueError("%d bits too small to represent integer value %d."
                         % (bits, integer))

    # Converting integer to array
    v = numpy.zeros(bits or size, numpy.bool_)
    for i in range(0, size):
        v[-(i+1)] = integer & 1
        integer >>= 1

    return v


def int_to_bit_vector_large(integer, bits=0):
    """
    Transform integer into a bit vector, optionally of a specific length.

    This function is the special form that can handle very large integers
    (>64bit).

    :raises ValueError: If ``bits`` specified is smaller than the required bits
        to represent the given ``integer`` value.

    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by the
        vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian).
    :rtype: numpy.ndarray[bool]

    """
    # Can't use math version because floating-point precision runs out after
    # about 2^48
    # -2 to remove length of '0b' string prefix
    size = len(bin(integer)) - 2

    if bits and (bits - size) < 0:
        raise ValueError("%d bits too small to represent integer value %d."
                         % (bits, integer))

    # Converting integer to array
    v = numpy.zeros(bits or size, numpy.bool_)
    for i in range(0, size):
        v[-(i+1)] = integer & 1
        integer >>= 1

    return v


def popcount(v):
    """
    Count the number of bits set (number of 1-bits, not 0-bits).

    Pure python popcount algorithm adapted implementation at:
    see: https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    Maximum known stable value that can be passed through this method:
    2**256 - 2. See the ``popcount.v_max`` function property.

    :param v: Integer to count the set bits of. Must be a 32-bit integer or
        less.
    :type v: int

    :return: Number of set bits in the given integer ``v``.
    :rtype: int

    """
    # TODO: C implementation of this
    #       since this version, being in python, isn't faster than counting 1's
    #       in result of ``bin`` function.

    # Cannot take the log of 0.
    if not v:
        return 0

    # T is the number of bits used to represent v to the nearest power of 2
    ceil, log = math.ceil, math.log
    tp = max(8, int(2**ceil(log(v.bit_length()) / log(2))))
    t = 2**tp-1
    b = tp // 8

    # bit-length constrained
    h55 = t//3
    h33 = t//15*3
    h0f = t//255*15
    h01 = t//255

    # noinspection PyAugmentAssignment
    v = v - ((v >> 1) & h55)
    v = (v & h33) + ((v >> 2) & h33)
    v = (v + (v >> 4)) & h0f
    # Need the extra ``& t`` after the multiplication in order to simulate bit
    # truncation as if v were only a tp-bit integer
    # Magic 8 represents bits ina byte
    return ((v * h01) & t) >> ((b-1) * 8)

# Maximum known stable value that can be passed as ``v``.
popcount.v_max = (2**256) - 2
