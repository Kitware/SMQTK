__author__ = 'purg'


def partition_string(s, segments):
    """
    Partition a string into a number of segments. If the given number of
    segments does not divide evenly into the string's length, this function
    behaves erratically.

    This is useful when partitioning an MD5 sum of a file to determine where it
    should be placed in a directory tree system.

    :param s: String to partition.
    :type s: str

    :param segments: Number of segments to splut the string into
    :type segments: int

    :return: A list of N segments. If the given number of segments does not
        divide evenly into the string's length, this function behaves
        erratically.
    :rtype: list[str]

    """
    seg_len = len(s) // segments
    tail = len(s) % segments
    r = []
    for i in range(segments):
        r.append(s[i*seg_len:(i+1)*seg_len])
    if tail:
        r.append(s[-tail:])
    return r
