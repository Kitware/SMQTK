import abc
from collections import deque
import hashlib
import io
import logging
import mimetypes
import os
import os.path as osp
import tempfile

import six

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation import SmqtkRepresentation
from smqtk.utils import file_utils
from smqtk.utils import plugin


MIMETYPES = mimetypes.MimeTypes()


class DataElement (SmqtkRepresentation, plugin.Pluggable):
    """
    Abstract interface for a byte data container.

    The primary "value" of a ``DataElement`` is the byte content wrapped. Since
    this can technically change due to external forces, we cannot guarantee that
    an element is immutable. Thus ``DataElement`` instances are not considered
    generally hashable. Specific implementations may define a ``__hash__``
    method if that implementation reflects a data source that guarantees
    immutability.

    UUIDs should be cast-able to a string and maintain unique-ness after
    conversion.

    """

    @classmethod
    def from_uri(cls, uri):
        """
        Construct a new instance based on the given URI.

        This function may not be implemented for all DataElement types.

        :param uri: URI string to resolve into an element instance
        :type uri: str

        :raises NotImplementedError: This element type does not implement URI
            resolution.
        :raises smqtk.exceptions.InvalidUriError: This element type could not
            resolve the provided URI string.

        :return: New element instance of our type.
        :rtype: DataElement

        """
        raise NotImplementedError()

    def __init__(self):
        super(DataElement, self).__init__()
        self._temp_filepath_stack = []

    # Because we can't generally guarantee external data immutability.
    __hash__ = None

    def __del__(self):
        self.clean_temp()

    def __eq__(self, other):
        return isinstance(other, DataElement) and \
               self.get_bytes() == other.get_bytes()

    def __ne__(self, other):
        return not (self == other)

    @abc.abstractmethod
    def __repr__(self):
        return self.__class__.__name__

    def _write_new_temp(self, d):
        """
        Actually write our bytes to a new temp file
        Always creates new file.

        :param d: directory to write temp file in or None to use system default.
        :returns: path to file written

        """
        if d:
            file_utils.safe_create_dir(d)
        ext = MIMETYPES.guess_extension(self.content_type())
        # Exceptions because mimetypes is apparently REALLY OLD
        if ext in {'.jpe', '.jfif'}:
            ext = '.jpg'
        fd, fp = tempfile.mkstemp(
            suffix=ext or '',
            dir=d
        )
        os.close(fd)
        with open(fp, 'wb') as f:
            f.write(self.get_bytes())
        return fp

    def _clear_no_exist(self):
        """
        Clear paths in temp stack that don't exist on the system.
        """
        no_exist_paths = deque()  # tmp list of paths to remove
        for fp in self._temp_filepath_stack:
            if not osp.isfile(fp):
                no_exist_paths.append(fp)
        for fp in no_exist_paths:
            self._temp_filepath_stack.remove(fp)

    def md5(self):
        """
        Get the MD5 checksum of this element's binary content.

        :return: MD5 hex checksum of the data content.
        :rtype: str
        """
        return hashlib.md5(self.get_bytes()).hexdigest()

    def sha1(self):
        """
        Get the SHA1 checksum of this element's binary content.

        :return: SHA1 hex checksum of the data content.
        :rtype: str
        """
        return hashlib.sha1(self.get_bytes()).hexdigest()

    def sha512(self):
        """
        Get the SHA512 checksum of this element's binary content.

        :return: SHA512 hex checksum of the data content.
        :rtype: str
        """
        return hashlib.sha512(self.get_bytes()).hexdigest()

    def write_temp(self, temp_dir=None):
        """
        Write this data's bytes to a temporary file on disk, returning the path
        to the written file, whose extension is guessed based on this data's
        content type.

        It is not guaranteed that the returned file path does not point to the
        original data, i.e. writing to the returned filepath may modify the
        original data.

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

        # Clear out paths that don't exist.
        self._clear_no_exist()

        if temp_dir:
            abs_temp_dir = osp.abspath(osp.expanduser(temp_dir))
            # Check if dir is the base of any path in the current stack.
            for tf in self._temp_filepath_stack:
                if osp.dirname(tf) == abs_temp_dir:
                    return tf
            # nothing in stack with given base directory, create new temp file
            self._temp_filepath_stack.append(self._write_new_temp(temp_dir))

        elif not self._temp_filepath_stack:
            # write new temp file to platform specific temp directory
            self._temp_filepath_stack.append(self._write_new_temp(None))

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
        UUID for this data element.

        This many take different forms from integers to strings to a uuid.UUID
        instance. This must return a hashable data type.

        By default, this ends up being the hex stringification of the SHA1 hash
        of this data's bytes. Specific implementations may provide other UUIDs,
        however.

        :return: UUID value for this data element. This return value should be
            hashable.
        :rtype: collections.Hashable

        """
        # TODO(paul.tunison): Change to SHA512.
        return self.sha1()

    def to_buffered_reader(self):
        """
        Wrap this element's bytes in a ``io.BufferedReader`` instance for use as
        file-like object for reading.

        As we use the ``get_bytes`` function, this element's bytes must safely
        fit in memory for this method to be usable.

        :return: New BufferedReader instance
        :rtype: io.BufferedReader

        """
        return io.BufferedReader(io.BytesIO(self.get_bytes()))

    def is_read_only(self):
        """
        :return: If this element can only be read from.
        :rtype: bool
        """
        return not self.writable()

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
    def is_empty(self):
        """
        Check if this element contains no bytes.

        The intend of this method is to quickly check if there is any data
        behind this element, ideally without having to read all/any of the
        underlying data.

        :return: If this element contains 0 bytes.
        :rtype: bool

        """

    @abc.abstractmethod
    def get_bytes(self):
        """
        :return: Get the bytes for this data element.
        :rtype: bytes
        """

    @abc.abstractmethod
    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """

    @abc.abstractmethod
    def set_bytes(self, b):
        """
        Set bytes to this data element.

        Not all implementations may support setting bytes (check ``writable``
        method return).

        This base abstract method should be called by sub-class implementations
        first. We check for mutability based on ``writable()`` method return and
        invalidate checksum caches.

        :param b: bytes to set.
        :type b: str

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.

        """
        if not self.writable():
            raise ReadOnlyError("This %s element is read only." % self)


def get_data_element_impls(reload_modules=False):
    """
    Discover and return discovered ``DataElement`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``DATA_ELEMENT_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``DATA_ELEMENT_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``DataElement``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "DATA_ELEMENT_PATH"
    helper_var = "DATA_ELEMENT_CLASS"
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var,
                              DataElement, reload_modules=reload_modules)


def from_uri(uri, impl_generator=get_data_element_impls):
    """
    Create a data element instance from available plugin implementations.

    The first implementation that can resolve the URI is what is returned. If no
    implementations can resolve the URL, an ``InvalidUriError`` is raised.

    :param uri: URI to try to resolve into a DataElement instance.
    :type uri: str

    :param impl_generator: Function that returns a dictionary mapping
        implementation type names to the class type. By default this refers to
        the standard ``get_data_element_impls`` function, however this can be
        changed to refer to a custom set of classes if desired.
    :type impl_generator: () -> dict[str, type]

    :raises smqtk.exceptions.InvalidUriError: No data element implementations
        could resolve the given URI.

    :return: New data element instance providing access to the data pointed to
        by the input URI.
    :rtype: DataElement

    """
    log = logging.getLogger(__name__)
    log.debug("Trying to parse URI: '%s'", uri)

    #: :type: __generator[DataElement]
    de_type_iter = six.itervalues(impl_generator())
    inst = None
    for de_type in de_type_iter:
        try:
            inst = de_type.from_uri(uri)
        except NotImplementedError:
            pass
        except InvalidUriError as ex:
            log.debug("Implementation '%s' failed to parse URI: %s",
                      de_type.__name__, ex.reason)
        if inst is not None:
            break
    if inst is None:
        # TODO: Assume final fallback of FileElement?
        #       Since any string could be a file?
        raise InvalidUriError(uri, "No available implementation to handle URI.")
    return inst
