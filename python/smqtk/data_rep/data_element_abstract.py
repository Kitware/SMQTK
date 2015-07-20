__author__ = 'purg'

import abc
import hashlib
import logging
import mimetypes
import os
import os.path as osp
import tempfile

from smqtk.utils import safe_create_dir


MIMETYPES = mimetypes.MimeTypes()


class DataElement (object):
    """
    Base abstract class for a data element.

    Basic data elements have a UUID, some byte content, and a content type.

    DataElement implementations should be picklable for serialization, as some
    DataSet implementations will desire such functionality.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._md5_cache = None
        self._sha1_cache = None
        self._temp_filepath_stack = []

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __hash__(self):
        return hash(self.uuid())

    def __del__(self):
        self.clean_temp()

    # TODO: __eq__/__ne__ methods?

    def md5(self):
        """
        :return: MD5 hex string of the data content.
        :rtype: str
        """
        if not self._md5_cache:
            self._md5_cache = hashlib.md5(self.get_bytes()).hexdigest()
        return self._md5_cache

    def sha1(self):
        """
        :return: SHA1 hex string of the data content.
        :rtype: str
        """
        if not self._sha1_cache:
            self._sha1_cache = hashlib.sha1(self.get_bytes()).hexdigest()
        return self._sha1_cache

    def write_temp(self, temp_dir=None):
        """
        Write this data's bytes to a temporary file on disk, returning the path
        to the written file, whose extension is guessed based on this data's
        content type.

        NOTE:
            The file path returned should not be explicitly removed by the user.
            Instead, the ``clean_temp()`` method should be called on this
            object.

        :param temp_dir: Optional directory to write temporary file in,
            otherwise we use the platform default temporary files directory.
            If this is an empty string, we count it the same as having provided
            None.
        :type temp_dir: None or str

        :return: Path to the temporary file
        :rtype: str

        """
        # Write a new temp file if there aren't any in the stack, or if the none
        # of the entries' base directory is the provided temp_dir (when one is
        # provided).

        def write_temp(d):
            """ Returns path to file written. Always creates new file. """
            if d:
                safe_create_dir(d)
            ext = MIMETYPES.guess_extension(self.content_type())
            # Exceptions because mimetypes is apparently REALLY OLD
            if ext in {'.jpe', '.jfif'}:
                ext = '.jpg'
            fd, fp = tempfile.mkstemp(
                suffix=ext,
                dir=d
            )
            os.close(fd)
            with open(fp, 'wb') as f:
                f.write(self.get_bytes())
            return fp

        if temp_dir:
            abs_temp_dir = osp.abspath(osp.expanduser(temp_dir))
            # Check if dir is the base of any path in the current stack.
            for tf in self._temp_filepath_stack:
                if osp.dirname(tf) == abs_temp_dir:
                    return tf
            # nothing in stack with given base directory, create new temp file
            self._temp_filepath_stack.append(write_temp(temp_dir))

        elif not self._temp_filepath_stack:
            # write new temp file to platform specific temp directory
            self._temp_filepath_stack.append(write_temp(None))

        # return last written temp file.
        return self._temp_filepath_stack[-1]

    def clean_temp(self):
        """
        Clean any temporary files created by this element. This does nothing if
        no temporary files have been generated for this element yet.
        """
        if len(self._temp_filepath_stack):
            for fp in self._temp_filepath_stack:
                if os.path.isfile(fp):
                    os.remove(fp)
            self._temp_filepath_stack = []

    def uuid(self):
        """
        UUID for this data element. This many take different forms from integers
        to strings to a uuid.UUID instance. This must return a hashable data
        type.

        By default, this ends up being the stringification of the SHA1 hash of
        this data's bytes. Specific implementations may provide other UUIDs,
        however.

        :return: UUID value for this data element. This return value should be
            hashable.
        :rtype: collections.Hashable

        """
        return self.sha1()

    ###
    # Abstract methods
    #

    @abc.abstractmethod
    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return

    @abc.abstractmethod
    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return
