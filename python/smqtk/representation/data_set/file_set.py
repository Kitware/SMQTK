import cPickle
import os
import os.path as osp
import re

from smqtk.representation import DataElement, DataSet
from smqtk.utils import file_utils
from smqtk.utils.string_utils import partition_string


__author__ = "paul.tunison@kitware.com"


class DataFileSet (DataSet):
    """
    File-based data set

    File sets are initialized with a root directory, under which it finds and
    serialized DataElement instances. DataElement implementations are required
    to be picklable, so this is a valid assumption.

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

    def __init__(self, root_directory, uuid_chunk=10):
        """
        Initialize a new or existing file set from a root directory.

        :param root_directory: Directory that this file set is based in. For
            relative path resolution, see the ``work_relative`` parameter
            description.
        :type root_directory: str

        :param uuid_chunk: Number of segments to split data element UUID
            into when saving element serializations.
        :type uuid_chunk: int

        """
        self._root_dir = os.path.abspath(os.path.expanduser(root_directory))
        self._uuid_chunk = uuid_chunk

        self._log.debug("Initializing FileSet under root dir: %s",
                        self._root_dir)

    def _iter_file_tree(self):
        """
        Iterate over our file tree, yielding the file paths of serialized
            elements found.
        """
        for fp in file_utils.iter_directory_files(self._root_dir):
            m = self.SERIAL_FILE_RE.match(osp.basename(fp))
            if m:
                # if the path doesn't have the configured split chunking value,
                # it doesn't belong to this data set
                seg = osp.dirname(osp.relpath(fp, self._root_dir)).split(os.sep)
                if len(seg) == self._uuid_chunk:
                    yield fp

    def _uuid_from_fp(self, fp):
        return osp.dirname(osp.relpath(fp, self._root_dir)).replace(os.sep, '')

    def _containing_dir(self, uuid):
        """
        Return the containing directory for something with the given UUID value
        """
        return osp.join(self._root_dir,
                        *partition_string(uuid, self._uuid_chunk))

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
                yield cPickle.load(f)

    def get_config(self):
        return {
            "root_directory": self._root_dir,
            "uuid_chunk": self._uuid_chunk,
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
            assert isinstance(e, DataElement)
            uuid = str(e.uuid())
            fp = self._fp_for_uuid(uuid)
            file_utils.safe_create_dir(osp.dirname(fp))
            with open(fp, 'wb') as f:
                cPickle.dump(e, f)
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
        fp = self._fp_for_uuid(str(uuid))
        if not osp.isfile(fp):
            raise KeyError(uuid)
        else:
            with open(fp, 'rb') as f:
                return cPickle.load(f)


DATA_SET_CLASS = DataFileSet
