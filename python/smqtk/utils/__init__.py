"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""


def safe_create_dir(d):
    """
    Recursively create the given directory, ignoring the already-exists
    error if thrown.

    :param d: Directory filepath to create
    :type d: str

    :return: The directory that was create, i.e. the directory that was passed
        in (absolute form).
    :rtype: str

    """
    import os
    import errno
    d = os.path.abspath(os.path.expanduser(d))
    try:
        os.makedirs(d)
    except OSError, ex:
        if ex.errno == errno.EEXIST and os.path.exists(d):
            pass
        else:
            raise
    return d


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


###
# In specific ordering for dependency resolution
#

# No internal util dependencies
from .bin_utils import initialize_logging, SMQTKOptParser
from .database_info import DatabaseInfo
from .read_write_lock import ReaderUpdateException, DummyRWLock, ReadWriteLock
from .safe_config_comment_parser import SafeConfigCommentParser
from .signal_handler import SignalHandler
from .simple_timer import SimpleTimer

from .distance_kernel import DistanceKernel
from .feature_memory import FeatureMemory, FeatureMemoryMap
from .timed_cache import TimedCache
from .proxy_manager import ProxyManager
