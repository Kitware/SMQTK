import os
import os.path as osp
import re

try:
    from six.moves import cPickle as pickle
except ImportError:
    import pickle

from smqtk.representation import DataElement, DataSet
from smqtk.utils import file_utils
from smqtk.utils.string_utils import partition_string


class DataFileSet (DataSet):
    """
    File-based data set

    This implementation serializes each stored DataElement to a single file on
    disk. The only memory overhead for this implementation is the mapping of
    data element original UUID to the file path on disk.

    File sets are initialized with a root directory, under which it finds and
    serializes DataElement instances. DataElement implementations are required
    to be picklable, so this is a valid assumption.

    Serializations are saved/loaded with respect to a ``uuid_chunk`` property,
    which defines how files are split into subdirectories based on their UUID.
    This is helpful when a lot of files are being stored as too many files in a
    single directory generally slows down filesystem access to that directory.

    """
    # TODO: Use file-based locking mechanism to make thread/process safe

    # Filename template for serialized files. Requires template
    SERIAL_FILE_TEMPLATE = "UUID_%s.dataElement"

    # Regex for matching file names as valid FileSet serialized elements
    # - yields two groups, the first is the UUID, the second is the SHA1 sum
    SERIAL_FILE_RE = re.compile("UUID_(\w+)\.dataElement")

    @classmethod
    def is_usable(cls):
        """
        Check whether this data set implementations is available for use.

        This is always true for this implementation as there are no required 3rd
        party dependencies

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        return True

    def __init__(self, root_directory, uuid_chunk=10, pickle_protocol=-1):
        """
        Initialize a new or existing file set from a root directory.

        If ``uuid_chunk`` is set to None or 1 we look for and save file
        caches in a flat manner under the root directory given. If greater than
        1, we split ``DataElement`` UUIDs into that many segments and use all
        but the last segment to define what sub-directory to place the
        serialization under the given ``root_directory``. For example, if
        ``uuid_chunk`` is 3 and a file's UUID is "abcdef", we store the
        element's serialization under the directory "<root_directory>/ab/cd/".
        We don't use the last segment in order to allow multiple files, but not
        all of them, to exist in leaf directories as opposed to there being a
        separate directory for each file, which is excessive.

        :param root_directory: Directory that this file set is based in. For
            relative path resolution, see the ``work_relative`` parameter
            description.
        :type root_directory: str

        :param uuid_chunk: Number of segments to split data element UUID
            into when saving element serializations. This should be None or a
            positive integer.
        :type uuid_chunk: None | int

        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :type pickle_protocol: int

        """
        super(DataFileSet, self).__init__()
        if not (uuid_chunk is None or uuid_chunk > 0):
            raise ValueError('Uuid chunk must either be None or a positive '
                             'integer.')

        self._root_dir = os.path.expanduser(root_directory)
        self._uuid_chunk = uuid_chunk
        self.pickle_protocol = pickle_protocol

        self._log.debug("Initializing FileSet under root dir: %s",
                        self._root_dir)

    def _iter_file_tree(self):
        """
        Iterate over our file tree, yielding the file paths of serialized
        elements found in the expected sub-directories.
        """
        # Select how far we need to descent into root based on chunk level.
        recurse = (self._uuid_chunk and self._uuid_chunk - 1) or None

        for fp in file_utils.iter_directory_files(self._root_dir, recurse):
            m = self.SERIAL_FILE_RE.match(osp.basename(fp))
            if m:
                # Where file is under root to see if it is a file we care about
                # according to our ``uuid_chunk`` value.
                # - ``seg`` includes file name, so it will at least be of length
                #   1, so its unmodified length should equal ``uuid_chunk``.
                # - Exceptions: len(seg) == 1 and uuid_chunk in {None, 0, 1}
                seg = osp.relpath(fp, self._root_dir).split(os.sep)
                if self._uuid_chunk in {None, 1} and len(seg) == 1:
                    yield fp
                elif len(seg) == self._uuid_chunk:
                    yield fp

    def _containing_dir(self, uuid):
        """
        Return the containing directory for something with the given UUID value
        """
        if not self._uuid_chunk:
            # No sub-directory storage configured
            return self._root_dir

        str_uuid = str(uuid)
        # TODO(paul.tunison): Modify uuid string if to short for set UUID chunk.
        #     e.g. if uuid is the integer 1 and chunk size is 10, we should
        #     convert strigified result to be at least length 10?
        #     Do this in _fp_for_uuid method?
        leading_parts = partition_string(str_uuid, self._uuid_chunk)[:-1]
        return osp.join(self._root_dir, *leading_parts)

    def _fp_for_uuid(self, uuid):
        """
        Return the filepath to where an element with the given UUID would be
        saved.
        """
        return osp.join(self._containing_dir(uuid),
                        self.SERIAL_FILE_TEMPLATE % uuid)

    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in no
            particular order.
        """
        for fp in self._iter_file_tree():
            # deserialize and yield
            with open(fp) as f:
                yield pickle.load(f)

    def get_config(self):
        return {
            "root_directory": self._root_dir,
            "uuid_chunk": self._uuid_chunk,
            "pickle_protocol": self.pickle_protocol,
        }

    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        c = 0
        for _ in self._iter_file_tree():
            c += 1
        return c

    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """
        s = set()
        for de in self:
            s.add(de.uuid())
        return s

    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the type
            that the set implementation expects or cares about.

        :return: True if the given uuid matches an element in this set, or False
            if it does not.
        :rtype: bool

        """
        # Try to access the expected file path like a hash table
        return osp.isfile(self._fp_for_uuid(uuid))

    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        :param elems: Data element(s) to add
        :type elems: list[smqtk.representation.DataElement]

        """
        for e in elems:
            assert isinstance(e, DataElement), \
                "Not given a DataElement for addition: '%s'" % e
            uuid = str(e.uuid())
            fp = self._fp_for_uuid(uuid)
            file_utils.safe_create_dir(osp.dirname(fp))
            with open(fp, 'wb') as f:
                pickle.dump(e, f, self.pickle_protocol)
            self._log.debug("Wrote out element %s", e)

    def get_data(self, uuid):
        """
        Get the data element the given uuid references, or raise an
        exception if the uuid does not reference any element in this set.

        :raises KeyError: If the given uuid does not refer to an element in
            this data set.

        :param uuid: The uuid of the element to retrieve.

        :return: The data element instance for the given uuid.
        :rtype: smqtk.representation.DataElement

        """
        fp = self._fp_for_uuid(uuid)
        if not osp.isfile(fp):
            raise KeyError(uuid)
        else:
            with open(fp, 'rb') as f:
                return pickle.load(f)


DATA_SET_CLASS = DataFileSet
