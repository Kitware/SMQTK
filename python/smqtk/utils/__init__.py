"""
LICENCE
-------
Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import functools
import operator as op

# noinspection PyUnresolvedReferences
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
    numer = functools.reduce(op.mul, range(n, n - r, -1))
    denom = functools.reduce(op.mul, range(1, r + 1))
    return numer // denom


###
# In specific ordering for dependency resolution
#

# No internal util dependencies
from .base_object import SmqtkObject
from .bin_utils import initialize_logging
from .content_type_validator import ContentTypeValidator
from .database_info import DatabaseInfo
from .iter_validation import check_empty_iterable
from .read_write_lock import ReaderUpdateException, DummyRWLock, ReadWriteLock
from .safe_config_comment_parser import SafeConfigCommentParser
from .signal_handler import SignalHandler
from .simple_timer import SimpleTimer
