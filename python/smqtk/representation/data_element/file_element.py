
import mimetypes
import os.path as osp

from smqtk.representation import DataElement

try:
    from tika import detector as tika_detector
except ImportError:
    tika_detector = None


__author__ = "paul.tunison@kitware.com"


# Fix global MIMETYPE map
if '.jfif' in mimetypes.types_map:
    del mimetypes.types_map['.jfif']
if '.jpe' in mimetypes.types_map:
    del mimetypes.types_map['.jpe']


class DataFileElement (DataElement):
    """
    File-based data element
    """

    @classmethod
    def is_usable(cls):
        # No dependencies
        return True

    def __init__(self, filepath):
        """
        Create a new FileElement.

        :param filepath: Path to the file to wrap.  If relative, it is
            interpreted as relative to the current working directory.
        :type filepath: str

        """
        super(DataFileElement, self).__init__()

        # Just expand a user-home `~` if present, keep relative if given.
        self._filepath = osp.expanduser(filepath)

        self._content_type = None
        if tika_detector:
            try:
                self._content_type = tika_detector.from_file(filepath)
            except IOError, ex:
                self._log.warn("Failed tika.detector.from_file content type "
                               "detection (error: %s), falling back to file "
                               "extension",
                               str(ex))
        # If no tika detector or it failed for some reason
        if not self._content_type:
            self._content_type = mimetypes.guess_type(filepath)[0]

    def __repr__(self):
        return super(DataFileElement, self).__repr__()[:-1] + \
            ", filepath: %s}" % self._filepath

    def get_config(self):
        return {
            "filepath": self._filepath
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return self._content_type

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

        For FileElement instances, this returns the original data file's path
        unless a `temp_dir` is specified that is not the directory that contains
        the original file.

        :param temp_dir: Optional directory to write temporary file in,
            otherwise we use the platform default temporary files directory.
        :type temp_dir: None or str

        :return: Path to the temporary file
        :rtype: str

        """
        if temp_dir:
            abs_temp_dir = osp.abspath(osp.expanduser(temp_dir))
            if abs_temp_dir != osp.dirname(self._filepath):
                return super(DataFileElement, self).write_temp(temp_dir)
        return self._filepath

    def clean_temp(self):
        """
        Clean any temporary files created by this element. This does nothing if
        no temporary files have been generated for this element.

        For FileElement instance's this does nothing as the ``write_temp()``
        method doesn't actually write any files.
        """
        # does the right thing regardless of what happened in write_temp
        return super(DataFileElement, self).clean_temp()


DATA_ELEMENT_CLASS = DataFileElement
