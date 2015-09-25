import csv
import os
import numpy
import re
import tempfile


__author__ = "paul.tunison@kitware.com"


def make_tempfile(*args, **kwds):
    """
    Wrapper for ``tempfile.mkstemp`` that closes/discards the file descriptor
    returned from the method. Arguments and keywords passed are given to
    ``tempfile.mkstemp``.

    :return: Path to a new user-owned temporary file.
    :rtype: str

    """
    fd, fp = tempfile.mkstemp(*args, **kwds)
    os.close(fd)
    return fp


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


def exclusive_touch(file_path):
    """
    Attempt to touch a file. If that file already exists, we return False.
    If the file was touched and created, we return True. Other OSErrors
    thrown beside the expected "file already exists" error are passed
    upwards.

    :param file_path: Path to the file to touch.
    :type file_path: str

    :return: True if we touched/created the file, false if we couldn't
    :rtype: bool

    """
    try:
        fd = os.open(file_path, os.O_CREAT | os.O_EXCL)
        os.close(fd)
        return True
    except OSError, ex:
        if ex.errno == 17:  # File exists, could not touch.
            return False
        else:
            raise


def iter_svm_file(filepath, width):
    """
    Iterate parsed vectors in a parsed "*.svm" file that encodes a sparce
    matrix, where each line consists of multiple "index:value" pairs in index
    order. Multiple lines construct a matrix.

    :param filepath: Path to the SVM file encoding an array per line
    :type filepath: str
    :param width: The known number of columns in the sparse vectors.
    :param width: int

    :return: Generator yielding ndarray vectors
    :rtype: collections.Iterable[numpy.core.multiarray.ndarray]

    """
    idx_val_re = re.compile("([0-9]+):([-+]?[0-9]*\.?[0-9]*)")
    with open(filepath, 'r') as infile:
        for line in infile:
            v = numpy.zeros(width, dtype=float)
            for seg in line.strip().split(' '):
                m = idx_val_re.match(seg)
                assert m is not None, \
                    "Invalid index:value match for segment '%s'" % seg
                idx, val = int(m.group(1)), float(m.group(2))
                v[idx] = val
            yield v


def iter_csv_file(filepath):
    """
    Iterate parsed vectors in a "*.csv" file that encodes descriptor output
    where each line is a descriptor vector. Multiple lines construct a matrix.

    :param filepath: Path to the CSV file encoding an array per line.
    :type filepath: str

    :return: Generator yielding ndarray vectors
    :rtype: collections.Iterable[numpy.core.multiarray.ndarray]

    """
    with open(filepath) as f:
        r = csv.reader(f)
        for l in r:
            yield numpy.array(l, dtype=float)