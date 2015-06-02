__author__ = 'purg'

import abc
import logging
import mimetypes
import os
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

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __hash__(self):
        return hash(self.uuid())

    def __del__(self):
        self.clean_temp()

    # TODO: __eq__/__ne__ methods?

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
        :type temp_dir: None or str

        :return: Path to the temporary file
        :rtype: str

        """
        if not hasattr(self, '_temp_filepath') or not self._temp_filepath:
            if temp_dir:
                safe_create_dir(temp_dir)
            # noinspection PyAttributeOutsideInit
            fd, self._temp_filepath = tempfile.mkstemp(
                suffix=MIMETYPES.guess_extension(self.content_type()),
                dir=temp_dir
            )
            os.close(fd)
            with open(self._temp_filepath, 'wb') as ofile:
                ofile.write(self.get_bytes())
        return self._temp_filepath

    def clean_temp(self):
        """
        Clean any temporary files created by this element. This does nothing if
        no temporary files have been generated for this element.
        """
        if hasattr(self, "_temp_filepath"):
            os.remove(self._temp_filepath)
            # noinspection PyAttributeOutsideInit
            self._temp_filepath = None

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
    def md5(self):
        """
        :return: MD5 hex string of the data content.
        :rtype: str
        """
        return

    @abc.abstractmethod
    def uuid(self):
        """
        UUID for this data element. This many take different forms from integers
        to strings to a uuid.UUID instance. This must return a hashable data
        type.

        :return: UUID value for this data element. This return value should be
            hashable.
        :rtype: collections.Hashable

        """
        return

    @abc.abstractmethod
    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return
