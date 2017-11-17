import csv
import errno
import os
import numpy
import re
import tempfile
import threading
import time

from smqtk.utils import SmqtkObject


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
    tmp_dir = file_dir if tmp_dir is None else tmp_dir
    fd, fp = tempfile.mkstemp(suffix=file_ext, prefix=file_base + '.',
                              dir=tmp_dir)
    try:
        c = os.write(fd, b)
        if c != len(b):
            raise RuntimeError("Failed to write all bytes to file.")
    except:
        # Remove temporary file if something bad happens.
        os.remove(fp)
        raise
    finally:
        os.close(fd)
    os.rename(fp, path)


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
    :rtype: collections.Iterable[str]

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
    import tika.detector
    return tika.detector.from_file(filepath)


class FileModificationMonitor (SmqtkObject, threading.Thread):
    """
    Utility object for triggering a callback function when an observed file
    changes based on file modification times observed
    """

    STATE_WAITING = 0   # Waiting for file to be modified
    STATE_WATCHING = 1  # Waiting for file to settle
    STATE_SETTLED = 2   # File has stopped being modified for settle period

    def __init__(self, filepath, monitor_interval, settle_window, callback):
        """
        On a separate thread, monitor the modification time of the file at the
        given file path. When the file is updated (after the file has stopped
        changing), trigger the provided callback function, given the monitored
        file path and the file stat event.

        We initially set ourselves as a daemon as that is the most probable
        usage of our functionality.

        :param filepath: Path to the file to monitor
        :type filepath: str

        :param monitor_interval: Frequency in seconds at which we check file
            modification times. This must be >= 0.
        :type monitor_interval: float

        :param settle_window: If a recently modified file is not modified again
            for this many seconds in a row, we consider the file done being
            modified and trigger the triggers ``callback``. This must be >= 0
            and should be >= the ``monitor_interval``.
        :type settle_window: float

        :param callback: Callback function that will be triggered every time
            the provided file has been updated and the settle time has safely
            expired.
        :type callback: (str) -> None

        :raises ValueError: The given filepath did not point to an existing,
            valid file.

        """
        SmqtkObject.__init__(self)
        threading.Thread.__init__(self, name=self.__class__.__name__)

        self.daemon = True

        self.filepath = filepath
        self.monitor_interval = monitor_interval
        self.settle_window = settle_window
        self.callback = callback

        self.event_stop = threading.Event()
        self.event_stop.set()  # make sure false

        self.state = self.STATE_WAITING

        if not os.path.isfile(self.filepath):
            raise ValueError("Provided filepath did not point to an existing, "
                             "valid file.")

        if monitor_interval < 0 or settle_window < 0:
            raise ValueError("Monitor and settle times must be >= 0")

    def stop(self):
        self._log.debug("stopped externally")
        self.event_stop.set()

    def stopped(self):
        return self.event_stop.is_set()

    def start(self):
        # Clear stop flag
        self.event_stop.clear()
        super(FileModificationMonitor, self).start()

    def run(self):
        # self._log.debug("starting run method")
        # mtime baseline
        last_mtime = os.path.getmtime(self.filepath)

        try:
            while not self.stopped():
                mtime = os.path.getmtime(self.filepath)

                # file has been updated
                if self.state == self.STATE_WAITING and last_mtime != mtime:
                    self.state = self.STATE_WATCHING
                    self._log.debug('change detected '
                                    '(mtime: %f -> %f, diff=%f) '
                                    ':: state(WAITING -> WATCHING)',
                                    last_mtime, mtime, mtime - last_mtime)

                # Wait until file is not being modified any more
                elif self.state == self.STATE_WATCHING:
                    t = time.time()
                    if t - mtime >= self.settle_window:
                        self.state = self.STATE_SETTLED
                        self._log.debug('file settled '
                                        '(mtime=%f, t=%f, diff=%f) '
                                        ':: state(WATCHING -> SETTLED)',
                                        mtime, t, t - mtime)
                    else:
                        self._log.debug('waiting for settle '
                                        '(mtime=%f, t=%f, diff=%f)...',
                                        mtime, t, t - mtime)
                        time.sleep(self.monitor_interval)

                elif self.state == self.STATE_SETTLED:
                    self.callback(self.filepath)
                    self.state = self.STATE_WAITING
                    self._log.debug('calling callback '
                                    ':: state(SETTLED -> WAITING)')

                # waiting for modification
                else:
                    # self._log.debug("waiting...")
                    time.sleep(self.monitor_interval)

                last_mtime = mtime
        finally:
            self.event_stop.set()

        self._log.debug('exiting')
