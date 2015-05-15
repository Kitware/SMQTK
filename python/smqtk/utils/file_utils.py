__author__ = 'purg'

import os


def iter_directory_files(d, recurse=True):
    """
    Iterates through files in the directory structure at the given directory.

    :param d: base directory path
    :type d: str

    :param recurse: If true, we recursively descend into directories in the
        given directory. If false, we only return the files in the given
        directory and not the sub-directories in the given directory.
    :type recurse: bool

    :return: Generator expression yielding absolute file paths under the given
        directory.
    :rtype: collections.Iterable[str]

    """
    d = os.path.abspath(d)
    for f in os.listdir(d):
        f = os.path.join(d, f)
        if os.path.isfile(f):
            yield f
        elif os.path.isdir(f):
            if recurse:
                for sf in iter_directory_files(f, recurse):
                    yield sf
        else:
            raise RuntimeError("Encountered something not a file or "
                               "directory? :: '%s'" % f)


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
