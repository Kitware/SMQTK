__author__ = 'purg'

import mimetypes
import os.path as osp

from smqtk_config import WORK_DIR
from smqtk.data_rep import DataElement


class DataFileElement (DataElement):
    """
    File-based data element
    """

    def __init__(self, filepath):
        """
        Create a new FileElement.

        :param filepath: Path to the file to wrap.
        :type filepath: str

        """
        super(DataFileElement, self).__init__()

        self._filepath = osp.abspath(osp.join(WORK_DIR,
                                              osp.expanduser(filepath)))
        self._content_type = mimetypes.guess_type(filepath)[0]

    def __repr__(self):
        return "%s{uuid: %s, md5: %s, filepath: %s}" \
               % (self.__class__.__name__, self.uuid(), self.md5(),
                  self._filepath)

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return self._content_type

    def uuid(self):
        """
        UUID for this data element.

        :return: UUID value for this data element.
        :rtype: str

        """
        return self.md5()

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return open(self._filepath, 'rb').read()

    def write_temp(self, temp_dir=None):
        """
        Write this data's bytes to a temporary file on disk, returning the path
        to the written file, whose extension is guessed based on this data's
        content type.

        NOTE:
            The file path returned should not be explicitly removed by the user.
            Instead, the ``clean_temp()`` method should be called on this
            object.

        For FileElement instances, this returns the original data file's path.

        :param temp_dir: Optional directory to write temporary file in,
            otherwise we use the platform default temporary files directory.
        :type temp_dir: None or str

        :return: Path to the temporary file
        :rtype: str

        """
        if temp_dir:
            return super(DataFileElement, self).write_temp(temp_dir=temp_dir)
        else:
            return self._filepath

    def clean_temp(self):
        """
        Clean any temporary files created by this element. This does nothing if
        no temporary files have been generated for this element.

        For FileElement instance's this does nothing as the ``write_temp()``
        method doesn't actually write any files.
        """
        return super(DataFileElement, self).clean_temp()


DATA_ELEMENT_CLASS = DataFileElement
