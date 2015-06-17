__author__ = 'purg'


def partition_string(s, segments):
    """
    Partition a string into a number of segments. If the given number of
    segments does not divide evenly into the string's length, extra characters
    are added to the leading segments in order to allow for the requested number
    of segments.

    This is useful when partitioning an MD5 sum of a file to determine where it
    should be placed in a directory tree system.

    >>> partition_string("foobar", 2)
    ['foo', 'bar']
    >>> partition_string("foobar", 4)
    ['fo', 'ob', 'a', 'r']
    >>> partition_string('d7ca25c5-b886-4a1b-87fe-5945313d350b', 11)
    ['d7ca', '25c5', '-b88', '6-4', 'a1b', '-87', 'fe-', '594', '531', '3d3', '50b']

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
