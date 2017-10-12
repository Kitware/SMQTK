import sys

import numpy

from . import ncr

try:
    from numba import jit

except (ImportError, TypeError):

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


def _bit_vector_to_int_fast(v):
    """
    Transform a numpy vector representing a sequence of binary bits [0 | 1]
    into an integer representation.

    This version works only on contiguous arrays.

    :param v: 1D Vector of bits
    :type v: numpy.core.multiarray.ndarray

    :return: Integer equivalent
    :rtype: int
    """
    pad = '\x00' * (v.dtype.itemsize - 1)
    if (
            v.dtype.byteorder == '<' or
            (v.dtype.byteorder == '=' and sys.byteorder == 'little')):
        zero_str = '\x00{}'.format(pad)
        one_str = '\x01{}'.format(pad)
    else:
        zero_str = '{}\x00'.format(pad)
        one_str = '{}\x01'.format(pad)
    return int(str(v.data).replace(one_str, '1').replace(zero_str, '0'), 2)


@jit
def bit_vector_to_int(v):
    """
    Transform a numpy vector representing a sequence of binary bits [0 | 1]
    into an integer representation.

    This version handles vectors of up to 64bits in size.

    :param v: 1D Vector of bits
    :type v: numpy.core.multiarray.ndarray

    :return: Integer equivalent
    :rtype: int

    """
    if not hasattr(v, 'data') or v.dtype == numpy.dtype('object'):
        c = 0L
        for b in v:
            c = (c * 2L) + int(b)
        return c
    else:
        return _bit_vector_to_int_fast(v)


def bit_vector_to_int_large(v):
    """
    Transform a numpy vector representing a sequence of binary bits [0 | 1]
    into an integer representation.

    This function is the special form that can handle very large integers
    (>64bit).

    :param v: 1D Vector of bits
    :type v: numpy.core.multiarray.ndarray

    :return: Integer equivalent
    :rtype: int

    """
    if not hasattr(v, 'data') or v.dtype == numpy.dtype('object'):
        return sum(int(x) << i for i, x in enumerate(v[::-1]))
    else:
        return _bit_vector_to_int_fast(v)


def _int_to_bit_vector_fast(integer, bits=0):
    """
    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by
        the vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian), or None if too few
        ``bits`` were specified to contain the result.
    :rtype: numpy.ndarray[bool]

    """
    brep = bin(integer)
    rep_len = len(brep) - 2
    if bits and (bits - rep_len) < 0:
        return None
    brep = brep.replace('0b', '\x00' * (bits - rep_len))
    brep = brep.replace('0', '\x00').replace('1', '\x01')
    return numpy.frombuffer(brep, dtype='bool')


@jit
def int_to_bit_vector(integer, bits=0):
    """
    Transform integer into a bit vector, optionally of a specific length.

    This version handles vectors of up to 64bits in size.

    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by
        the vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian), or None if too few
        ``bits`` were specified to contain the result.
    :rtype: numpy.ndarray[bool]

    """
    return _int_to_bit_vector_fast(integer, bits)


def int_to_bit_vector_large(integer, bits=0):
    """
    Transform integer into a bit vector, optionally of a specific length.

    This function is the special form that can handle very large integers
    (>64bit).

    :param integer: integer to convert
    :type integer: int

    :param bits: Optional fixed number of bits that should be represented by
        the vector.
    :type bits: Optional specification of the size of returned vector.

    :return: Bit vector as numpy array (big endian), or None if too few
        ``bits`` were specified to contain the result.
    :rtype: numpy.ndarray[bool]

    """
    return _int_to_bit_vector_fast(integer, bits)


def popcount(v):
    """
    Pure python popcount algorithm adapted implementation at:
    see: https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    This is limited to 32-bit integer representation.

    """
    # TODO: C implementation of this
    #       since this version, being in python, isn't faster than above bin
    #       use
    if not v:
        return 0

    # T is the number of bits used to represent v to the nearest power of 2
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
