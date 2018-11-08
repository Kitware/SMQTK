import functools
import math
import operator as op

from six.moves import range


def ncr(n, r):
    """
    N-choose-r method, returning the number of combinations possible in integer
    form.

    From dheerosaur:
        http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python

    :param n: Selection pool size.
    :type n: int

    :param r: permutation selection size.
    :type r: int

    :return: Number of n-choose-r permutations for the given n and r.
    :rtype: int

    """
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
    # denom = functools.reduce(op.mul, range(1, r+1), 1)
    denom = math.factorial(r)
    return numer // denom


