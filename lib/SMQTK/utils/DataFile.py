"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import hashlib
import logging


class DataFile (object):
    """
    Basic data file representation
    """

    def __init__(self, filepath):
        """
        :param filepath: Path to the data file
        :type filepath: str
        """
        self._filepath = filepath

        # Cache variables
        self.__md5_cache = None

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @property
    def filepath(self):
        return self._filepath

    @property
    def md5sum(self):
        """
        :return: the MD5 sum of the data file
        :rtype: str
        """
        if self.__md5_cache is None:
            with open(self.filepath, 'rb') as data_file:
                self.__md5_cache = hashlib.md5(data_file.read()).hexdigest()
        return self.__md5_cache

    def split_md5sum(self, parts):
        """
        :param parts: Number of segments to split the MD5 sum into
        :type parts: int

        :return: A list of N mostly equal segments. If the number of parts
            cannot be evenly divided,
        :rtype: list of str

        """
        md5 = self.md5sum
        seg_len = len(md5) // parts
        tail = len(md5) % parts
        segments = []
        for i in xrange(parts):
            segments.append(md5[i*seg_len:(i+1)*seg_len])
        if tail:
            segments.append(md5[-tail:])
        return segments
