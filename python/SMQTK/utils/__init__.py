"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""


def safe_create_dir(d):
    """
    Recursively create the given directory, ignoring the already-exists error if
    thrown.

    :param d: Directory filepath to create
    :type d: str

    """
    import os
    import errno
    try:
        os.makedirs(d)
    except OSError, ex:
        if ex.errno == errno.EEXIST and os.path.exists(d):
            pass
        else:
            raise


def touch(fname):
    """
    Touch a file, creating it if it doesn't exist, setting its updated time to
    now.

    :param fname: File path to touch.
    :type fname: str

    """
    import os
    with open(fname, 'a'):
        os.utime(fname, None)


from .DatabaseInfo import DatabaseInfo
from .DataFile import DataFile
from .DataIngest import DataIngest
from .DistanceKernel import DistanceKernel
from .FeatureMemory import FeatureMemory, FeatureMemoryMap
from .ReadWriteLock import ReaderUpdateException, DummyRWLock, ReadWriteLock
from .SafeConfigCommentParser import SafeConfigCommentParser
from .SignalHandler import SignalHandler
from .SimpleTimer import SimpleTimer
from .TimedCache import TimedCache
from .VideoFile import VideoFile
from .VideoIngest import VideoIngest
