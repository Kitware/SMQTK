import csv
import errno
import os
import numpy
import re
import tempfile


__author__ = "paul.tunison@kitware.com"


def safe_create_dir(d):
    """
    Recursively create the given directory, ignoring the already-exists
    error if thrown.

    :param d: Directory filepath to create
    :type d: str

    :return: The directory that was created, i.e. the directory that was passed
        (in absolute form).
    :rtype: str

    """
    d = os.path.abspath(os.path.expanduser(d))
    try:
        os.makedirs(d)
    except OSError, ex:
        if ex.errno == errno.EEXIST and os.path.exists(d):
            pass
        else:
            raise
    return d


def make_tempfile(suffix="", prefix="tmp", dir=None, text=False):
    """
    Wrapper for ``tempfile.mkstemp`` that closes/discards the file descriptor
    returned from the method. Arguments/keywords passed are the same as, and
    passed directly to ``tempfile.mkstemp``.

    :return: Path to a new user-owned temporary file.
    :rtype: str

    """
    fd, fp = tempfile.mkstemp(suffix, prefix, dir, text)
    os.close(fd)
    return fp


def iter_directory_files(d, recurse=True):
    """
    Iterates through files in the directory structure at the given directory.

    :param d: base directory path
    :type d: str

    :param recurse: If true, we recursively descend into all directories under
        the given directory. If false, we only return the files in the given
        directory and not the sub-directories in the given directory. If this is
        an integer (positive), on only recurse that many sub-directories.
    :type recurse: bool | int

    :return: Generator expression yielding absolute file paths under the given
        directory.
    :rtype: collections.Iterable[str]

    """
    d = os.path.abspath(d)
    for dirpath, dirnames, filenames in os.walk(d):
        for fname in filenames:
            yield os.path.join(dirpath, fname)
        if not recurse:
            break
        if not recurse:
            break
        elif recurse is not True and dirpath != d:
            # must be an integer
            level = len(os.path.relpath(dirpath, d).split(os.sep))
            if level == recurse:
                # Empty directories descending
                del dirnames[:]
        # else recurse fully


def touch(fname):
    """
    Touch a file, creating it if it doesn't exist, setting its updated time to
    now.

    :param fname: File path to touch.
    :type fname: str

    """
    with open(fname, 'a'):
        os.utime(fname, None)


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