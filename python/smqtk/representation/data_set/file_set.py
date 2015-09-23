import cPickle
import multiprocessing
import os
import re

from smqtk.representation import DataElement, DataSet
from smqtk.utils import safe_create_dir
from smqtk.utils.file_utils import iter_directory_files
from smqtk.utils.string_utils import partition_string


__author__ = 'purg'


class DataFileSet (DataSet):
    """
    File-based data set. Data elements will all be file-based (DataFile type,
    see ``../data_element/file_element.py``).

    File sets are initialized with a root directory, under which it attempts to
    find existing serialized DataElement pickle files. This notion means that
    DataElement implementations stored must be picklable.

    This implementation does not currently suppose asynchronous modification on
    separate processes as they will conflict with each other on what to write to
    disk. This implementation should, however, be thread safe.

    """

    # Filename template for serialized files. Requires template
    SERIAL_FILE_TEMPLATE = "UUID_%s.SHA1_%s.dataElement"

    # Regex for matching file names as valid FileSet serialized elements
    # - yields two groups, the first is the UUID, the second is the SHA1 sum
    SERIAL_FILE_RE = re.compile("UUID_(\w+).SHA1_(\w+).dataElement")

    def __init__(self, root_directory, sha1_chunk=10):
        """
        Initialize a new or existing file set from a root directory.

        :param root_directory: Directory that this file set is based in. For
            relative path resolution, see the ``work_relative`` parameter
            description.
        :type root_directory: str

        :param sha1_chunk: Number of segments to split data element SHA1 sum
            into when saving element serializations.
        :type sha1_chunk: int

        """
        self._root_dir = os.path.abspath(os.path.expanduser(root_directory))
        self._sha1_chunk = sha1_chunk

        self._log.debug("Initializing FileSet under root dir: %s",
                        self._root_dir)

        #: :type: dict[object, smqtk.representation.DataElement]
        self._element_map = {}
        self._element_map_lock = multiprocessing.RLock()

        self._discover_data_elements()

        # Flag for when new elements have been added to the data set, which
        # triggers a filesystem serialization dump upon instance garbage
        # collection.
        self._new_elem_added = False

    def __del__(self):
        """
        Serialize out element contents on deletion.
        """
        if self._new_elem_added:
            self._save_data_elements()

    def __contains__(self, d):
        """
        :param d: DataElement to test for containment
        :type d: smqtk.representation.DataElement

        :return: True of this DataSet contains the given data element.
        :rtype: bool

        """
        with self._element_map_lock:
            return d.uuid() in self._element_map

    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in UUID
            order, if sortable. If not, then in no particular order.
        """
        for k in sorted(self.uuids()):
            with self._element_map_lock:
                yield self._element_map[k]

    def get_config(self):
        return {
            "root_directory": self._root_dir,
            "sha1_chunk": self._sha1_chunk,
        }

    def _discover_data_elements(self):
        """
        From the set root directory, find serialized files, deserialize them and
        store in instance mapping.
        """
        if os.path.isdir(self._root_dir):
            self._log.debug("Root directory exists, finding existing data "
                            "elements...")
            with self._element_map_lock:
                for fpath in iter_directory_files(self._root_dir, True):
                    m = self.SERIAL_FILE_RE.match(os.path.basename(fpath))
                    if m:
                        with open(fpath) as f:
                            #: :type: smqtk.representation.DataElement
                            de = cPickle.load(f)
                        self._element_map[de.uuid()] = de
                self._log.debug("Found %d elements", len(self._element_map))
        else:
            self._log.debug("Root dir doesn't exist, can't have existing "
                            "elements")

    def _save_data_elements(self):
        """
        Serialize out data elements in mapping into the root directory.
        """
        with self._element_map_lock:
            self._log.debug("Serializing data elements into: %s",
                            self._root_dir)
            for uuid, de in self._element_map.iteritems():
                # Remove any temporary files an element may have generated
                de.clean_temp()

                sha1 = de.sha1()
                # Leaving off trailing chunk so that we don't have a single
                # directory per sha1-sum.
                containing_dir = \
                    os.path.join(self._root_dir,
                                 *partition_string(sha1, self._sha1_chunk))
                if not os.path.isdir(containing_dir):
                    safe_create_dir(containing_dir)

                output_fname = os.path.join(
                    containing_dir,
                    self.SERIAL_FILE_TEMPLATE % (str(uuid), sha1)
                )
                with open(output_fname, 'wb') as ofile:
                    cPickle.dump(de, ofile)
            self._log.debug("Serializing data elements -- Done")

    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        with self._element_map_lock:
            return len(self._element_map)

    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """
        with self._element_map_lock:
            return set(self._element_map.keys())

    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the type
            that the set implementation expects or cares about.

        :return: True if the given uuid matches an element in this set, or False
            if it does not.
        :rtype: bool

        """
        with self._element_map_lock:
            return uuid in self._element_map

    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        :param elems: Data element(s) to add
        :type elems: list[smqtk.representation.DataElement]

        """
        with self._element_map_lock:
            for e in elems:
                assert isinstance(e, DataElement)
                self._element_map[e.uuid()] = e
                self._new_elem_added = True

    def get_data(self, uuid):
        with self._element_map_lock:
            return self._element_map[uuid]


DATA_SET_CLASS = DataFileSet
