import random

# noinspection PyUnresolvedReferences
from six.moves import range


def partition_string(s, segments):
    """
    Partition a string into a number of segments. If the given number of
    segments does not divide evenly into the string's length, extra characters
    are added to the leading segments in order to allow for the requested number
    of segments.

    This is useful when partitioning the checksum of a file to determine where
    it should be placed in a directory tree system.

    >>> partition_string("foobar", 2)
    ['foo', 'bar']
    >>> partition_string("foobar", 4)
    ['fo', 'ob', 'a', 'r']
    >>> partition_string("foobar", 6)
    ['f', 'o', 'o', 'b', 'a', 'r']

    If the string is not evenly divisible by the requested number of segments,
    then the length of trailing segments will be shorter than leading segments.

    >>> partition_string('d7ca25c5-b886-4a1b-87fe-5945313d350b', 11)
    ['d7ca', '25c5', '-b88', '6-4', 'a1b', '-87', 'fe-', '594', '531', '3d3', '50b']
    >>> partition_string('abcde', 2)
    ['abc', 'de']
    >>> partition_string('abcde', 4)
    ['ab', 'c', 'd', 'e']

    If the number of segments is greater than the number of characters in the
    input string, an assertion error is raised.

    >>> partition_string('a', 2)
    Traceback (most recent call last):
        ...
    AssertionError: Cannot split given string into more segments than there are characters in the string!

    :raises AssertionError: Segmentation value greater than the length of the
        string.

    :param s: String to partition.
    :type s: str

    :param segments: Number of segments to split the string into
    :type segments: int

    :return: A list of N segments. If the given number of segments does not
        divide evenly into the string's length, this function behaves
        erratically.
    :rtype: list[str]

    """
    assert segments <= len(s), \
        "Cannot split given string into more segments than there are " \
        "characters in the string!"

    seg_len = len(s) // segments
    extra_iters = len(s) % segments

    r = []
    i = 0
    for k in range(segments):
        j = i + seg_len + (k < extra_iters)
        r.append(s[i:j])
        i = j

    return r


DEFAULT_CHAR_SET = \
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def random_characters(n, char_set=DEFAULT_CHAR_SET):
    """
    Return ``n`` random characters from the given ``char_set``.

    If ``n`` is a floating point valid, it is cast to an integer (floor).
    The default ``char_set`` includes a-z, A-Z and 0-9.

    :param n: Number of random characters to return.
    :type n: int

    :param char_set: Sequence of characters to pull from when constructing
        random sequence.
    :type char_set: str | unicode

    :return: New string of random characters of length ``n`` from the given
        ``char_set``.
    :rtype: str | unicode

    :raises ValueError: If ``char_set`` given is empty, or ``n`` is negative.

    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be a positive integer.")
    l = len(char_set)
    if l == 0:
        raise ValueError("Empty char_set given.")
    return ''.join(char_set[random.randint(0, l - 1)] for _ in range(n))
