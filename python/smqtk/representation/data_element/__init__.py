import abc
import hashlib
import logging
import mimetypes
import os
import os.path as osp
import tempfile

from smqtk.utils import safe_create_dir
from smqtk.utils.configurable_interface import Configurable


__author__ = "paul.tunison@kitware.com"


MIMETYPES = mimetypes.MimeTypes()


class DataElement (Configurable):
    """
    Abstract interface for a byte data.

    Basic data elements have a UUID, some byte content, a content type, and
    checksum accessor methods.

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

    def __eq__(self, other):
        return isinstance(other, DataElement) and \
               self.get_bytes() == other.get_bytes()

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s{uuid: %s, content_type: '%s'}" \
               % (self.__class__.__name__, self.uuid(), self.content_type())

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

    @abc.abstractmethod
    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """


def get_data_element_impls(reload_modules=False):
    """
    Discover and return DataElement implementation classes found in the plugin
    directory. Keys in the returned map are the names of the discovered classes
    and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with and alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module, we first look for a helper variable by the name
    ``DATA_ELEMENT_CLASS``, which can either be a single class object or an
    iterable of class objects, to be exported. If the variable is set to None,
    we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same na e and casing as the module's
    name. If neither are found, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class objects of type ``DataElement`` whose keys
        are the string names of the classes.
    :rtype: dict[str, type]

    """
    import os
    from smqtk.utils.plugin import get_plugins

    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "DATA_ELEMENT_CLASS"
    return get_plugins(__name__, this_dir, helper_var, DataElement, None,
                       reload_modules)
