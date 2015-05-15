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
