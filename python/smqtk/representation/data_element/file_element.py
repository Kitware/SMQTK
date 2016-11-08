import mimetypes
import os.path as osp
import re

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation import DataElement
from smqtk.utils.file_utils import safe_create_dir

try:
    import magic
    # We know there are multiple modules named magic. Make sure the function we
    # expect is there.
    # noinspection PyStatementEffect
    magic.detect_from_filename
except (ImportError, AttributeError):
    magic = None

try:
    from tika import detector as tika_detector
except ImportError:
    tika_detector = None


# Fix global MIMETYPE map.
# Default map maps "image/jpeg" with some rarely used extensions before the
# common ones. We remove them here so we don't encounter them.
if '.jfif' in mimetypes.types_map:
    del mimetypes.types_map['.jfif']
if '.jpe' in mimetypes.types_map:
    del mimetypes.types_map['.jpe']


class DataFileElement (DataElement):
    """
    File-based data element
    """

    # Regex for file paths optionally including the file:// and / prefix.
    # Allows any character between slashes currently.
    FILE_URI_RE = re.compile("^(?:file://)?(/?[^/]+(?:/[^/]+)*)$")

    @classmethod
    def is_usable(cls):
        # No dependencies
        return True

    @classmethod
    def from_uri(cls, uri):
        """
        Construct a new instance based on the given URI.

        File elements can resolve any URI that looks like an absolute or
        relative path, or if the URI explicitly has the "file://" header.

        When the "file://" header is used, we expect an absolute path, including
        the leading slash. This means it will look like there are 3 slashes
        after the "file:", for example: "file:///home/me/somefile.txt".

        If this is given a URI with what looks like another URI header (e.g.
        "base64://..."), we thrown an InvalidUriError. This ends up being due to
        the `//` component, which we treat as an invalid path, not because of
        any special parsing.

        :param uri: URI string to resolve into an element instance
        :type uri: str

        :raises smqtk.exceptions.InvalidUriError: This element type could not
            resolve the provided URI string.

        :return: New element instance of our type.
        :rtype: DataElement

        """
        path_match = cls.FILE_URI_RE.match(uri)

        # if did not match RE, then not a valid path to a file (i.e. had a
        # trailing slash, file:// prefix malformed, etc.)
        if path_match is None:
            raise InvalidUriError(uri, "Malformed URI")

        path = path_match.group(1)

        # When given the file:// prefix, the encoded path must be absolute
        # Stealing the notion based on how Google Chrome handles file:// URIs
        if uri.startswith("file://") and not osp.isabs(path):
            raise InvalidUriError(uri, "Found file:// prefix, but path was not "
                                       "absolute")

        return DataFileElement(path)

    def __init__(self, filepath, readonly=False):
        """
        Create a new FileElement.

        :param filepath: Path to the file to wrap.  If relative, it is
            interpreted as relative to the current working directory.
        :type filepath: str

        :param readonly: If this element should allow writing or not.
        :type readonly: bool

        """
        super(DataFileElement, self).__init__()

        # Just expand a user-home `~` if present, keep relative if that's what
        # was given.
        self._filepath = osp.expanduser(filepath)
        self._readonly = bool(readonly)

        self._content_type = None
        if magic and osp.isfile(filepath):
            d = magic.detect_from_filename(filepath)
            self._content_type = d.mime_type
        elif tika_detector:
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
        return super(DataFileElement, self).__repr__() + \
            "{filepath: %s, readonly: %s}" % (self._filepath, self._readonly)

    def get_config(self):
        return {
            "filepath": self._filepath,
            "readonly": self._readonly,
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        return self._content_type

    def is_empty(self):
        """
        Check if this element contains no bytes.

        This plugin checks if the file on disk is greater than 0 in size.

        :return: If this element contains 0 bytes.
        :rtype: bool

        """
        return osp.exists(self._filepath) and osp.getsize(self._filepath) > 0

    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return open(self._filepath, 'rb').read()

    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        return not self._readonly

    def set_bytes(self, b):
        """
        Set bytes to this data element in the form of a string.

        Not all implementations may support setting bytes (writing). See the
        ``writable`` method.

        :param b: bytes to set.
        :type b: str

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.

        """
        if not self._readonly:
            # Make sure containing directory exists
            safe_create_dir(osp.dirname(self._filepath))
            with open(self._filepath) as f:
                f.write(b)
        else:
            raise ReadOnlyError("This file element is read only.")

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
            # Checking that the dir given isn't where the filepath lives
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
