import csv
import errno
import os
import numpy
import re
import sys
import tempfile
import warnings


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
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.exists(d):
            pass
        else:
            raise
    return d


def safe_file_write(path, b, tmp_dir=None):
    """
    Safely write to a file in such a way that the target file is never
    incompletely written to due to error or multiple agents attempting to
    writes.

    We leverage that most OSs have an atomic move/rename operation by first
    writing bytes to a separate temporary file first, then renaming the
    temporary file to the final destination path when write is complete.
    Temporary files are written to the same directory as the target file unless
    otherwise specified.

    **NOTE:** *Windows does not have an atomic file rename and this function
    currently does not do anything special to ensure atomic rename on Windows.*

    :param path: Path to the file to write to.
    :type path: str

    :param b: Byte iterable to write to file.
    :type b: str | bytes

    :param tmp_dir: Optional custom directory to write the intermediate
        temporary file to. This directory must already exist.
    :type tmp_dir: None | str

    """
    file_dir = os.path.dirname(path)
    file_name = os.path.basename(path)
    file_base, file_ext = os.path.splitext(file_name)

    # Make sure containing directory exists
    safe_create_dir(file_dir)

    # Write to a temporary file first, then OS move the temp file to the final
    # destination. This is due to, on most OSs, a file rename/move being atomic.
    # TODO(paul.tunison): Do something else on windows since moves there are not
    #   guaranteed atomic.
    if sys.platform == 'win32':
        warnings.warn("``safe_file_write`` attempts an OS rename operation. "
                      "This is not atomic on a Windows platform.")
    tmp_dir = file_dir if tmp_dir is None else tmp_dir
    f = tempfile.NamedTemporaryFile(suffix=file_ext, prefix=file_base + '.',
                                    dir=tmp_dir, delete=False)
    try:
        with f:
            f.write(b)
            # TODO: If we find issues with files not being completely written
            #       to disk, we may have to perform a ``f.file.flish()`` with
            #       an ``os.fsync(f.file.fileno())`` to force a full flush to
            #       disk.
    except Exception:
        # Remove temporary file if anything bad happens.
        os.remove(f.name)
        raise
    os.rename(f.name, path)


def make_tempfile(suffix="", prefix="tmp", directory=None, text=False):
    """
    Wrapper for ``tempfile.mkstemp`` that closes/discards the file descriptor
    returned from the method. Arguments/keywords passed are the same as, and
    passed directly to ``tempfile.mkstemp``.

    :return: Path to a new user-owned temporary file.
    :rtype: str

    """
    fd, fp = tempfile.mkstemp(suffix, prefix, directory, text)
    os.close(fd)
    return fp


def iter_directory_files(d, recurse=True):
    """
    Iterates through files in the structure under the given directory.

    :param d: base directory path
    :type d: str

    :param recurse: If true, we recursively descend into all directories under
        the given directory. If false, we only return the files in the given
        directory and not the sub-directories in the given directory. If this is
        an integer (positive), on only recurse that many sub-directories.
    :type recurse: bool | int

    :return: Generator expression yielding absolute file paths under the given
        directory.
    :rtype: collections.abc.Iterable[str]

    """
    d = os.path.abspath(d)
    for dirpath, dirnames, filenames in os.walk(d):
        for fname in filenames:
            yield os.path.join(dirpath, fname)
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
    except OSError as ex:
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
    :rtype: collections.abc.Iterable[numpy.core.multiarray.ndarray]

    """
    idx_val_re = re.compile(r"([0-9]+):([-+]?[0-9]*\.?[0-9]*)")
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
    :rtype: collections.abc.Iterable[numpy.core.multiarray.ndarray]

    """
    with open(filepath) as f:
        r = csv.reader(f)
        for l in r:
            yield numpy.array(l, dtype=float)


def file_mimetype_filemagic(filepath):
    """
    Determine file mimetype using the file-magic module.

    The file the given path refers to must exist.

    :raises ImportError: ``magic`` python module not available.
    :raises IOError: ``filepath`` did not refer to an existing file.

    :param filepath: Path to the (existing) file to determine the mimetype of.
    :type filepath: str

    :return: MIMETYPE string identifier.
    :rtype: str

    """
    # noinspection PyUnresolvedReferences
    import magic
    if os.path.isfile(filepath):
        d = magic.detect_from_filename(filepath)
        return d.mime_type
    elif os.path.isdir(filepath):
        raise IOError(21, "Is a directory: '%s'" % filepath)
    else:
        raise IOError(2, "No such file or directory: '%s'" % filepath)


def file_mimetype_tika(filepath):
    """
    Determine file mimetype using ``tika`` module.

    The file the given path refers to must exist. This function may fail under
    multiprocessing situations.

    :raises ImportError: ``tika`` python module not available.
    :raises IOError: ``filepath`` did not refer to an existing file.

    :param filepath: Path to the (existing) file to determine the mimetype of.
    :type filepath: str

    :return: MIMETYPE string identifier.
    :rtype: str

    """
    # noinspection PyUnresolvedReferences
    import tika.detector
    return tika.detector.from_file(filepath)
