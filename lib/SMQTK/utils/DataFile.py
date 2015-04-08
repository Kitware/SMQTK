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

    This base-class assumes image data.

    """

    def __init__(self, filepath, uid=None):
        """
        :param filepath: Path to the data file
        :type filepath: str
        """
        self._filepath = filepath

        # When not part of an ingest, this is None, otherwise the value is its
        # integer unique ID in ingest
        self._uid = uid

        # Cache variables
        self._md5_cache = None

    def __hash__(self):
        return hash(self.md5sum)

    def __eq__(self, other):
        return isinstance(other, DataFile) and self.md5sum == other.md5sum

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s{uid: %s, md5: %s}" % (
            self.__class__.__name__, self.uid, self.md5sum
        )

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__))
                                 + "::" + self.md5sum)

    @property
    def filepath(self):
        return self._filepath

    @property
    def uid(self):
        """
        When not part of an ingest, this is None, otherwise the value is its
        integer unique ID in ingest.

        :return: UID in the ingest that contains this data file, or None of this
            data file is not part of an ingest.
        :rtype: int

        """
        return self._uid

    @property
    def md5sum(self):
        """
        :return: the MD5 sum of the data file
        :rtype: str
        """
        if self._md5_cache is None:
            with open(self.filepath, 'rb') as data_file:
                self._md5_cache = hashlib.md5(data_file.read()).hexdigest()
        return self._md5_cache

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

    def get_preview_image(self):
        """
        :return: The path to a preview image for this data file.
        :rtype: str
        """
        return self.filepath
