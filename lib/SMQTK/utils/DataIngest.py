"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging
import multiprocessing
import os
import os.path as osp
import re
import shutil

from SMQTK.utils import DataFile, safe_create_dir, touch


class DataIngest (object):
    """
    SMQTK data file ingest representation

    Data files stored lose their original filenames and we instead use the
    unique MD5 sum of the file.

    Within the specified data directory:
        data/
            content/
                <ingest files>
            explicit_ids.txt

    """

    # DataIngest data file type. This should be set to a class or factory method
    # that takes up to two arguments: the filepath and an optional UID setting.
    DATA_FILE_TYPE = DataFile

    # Files saved with MD5sum and integer ID
    FILE_TEMPLATE = "uid_%d.%s%s"  # (uid, md5, ext)
    FILE_REGEX = re.compile("uid_(\d+)\.(\w+)(\..*)")  # uid, md5, ext

    def __init__(self, data_dir, work_dir, starting_index=0):
        """
        :param data_dir: Base directory for data storage
        :type data_dir: str

        :param work_dir: Base directory for work storage
        :type work_dir: str

        :param starting_index: Starting index for added data files
        :type starting_index: int

        """
        self._data_dir = data_dir
        self._work_dir = work_dir
        self._next_id = starting_index

        # Explicit content management
        self._eid_list_file = osp.join(self.data_directory,
                                       "explicit_ids.txt")
        if not osp.isfile(self._eid_list_file):
            touch(self._eid_list_file)
        self._eid_set = set()
        self._eid_lock = multiprocessing.RLock()

        # Map of ID-to-file
        #: :type: dict of (int, DataFile)
        self._id_data_map = {}
        # Reverse mapping for reverse fetch
        # Value is a set of UID as files of the same MD5 may bhe
        #: :type: dict of (str, int)
        self._md5_id_map = {}

        self._load_existing_ingest()

    def __len__(self):
        return len(self._id_data_map)

    def _load_existing_ingest(self):
        """
        Update state given existing ingest at the known data directory
        """
        max_uid = 0
        for filepath in self._iter_ingest_files(self.content_directory):
            m = self.FILE_REGEX.match(os.path.basename(filepath))
            if m:
                uid, md5, ext = m.groups()
                uid = int(uid)
                self._id_data_map[uid] = self.DATA_FILE_TYPE(filepath, uid=uid)
                # If we allowed multiple of the same MD5...
                # if md5 not in self._md5_id_map:
                #     self._md5_id_map[md5] = set()
                # self._md5_id_map[md5].add(uid)
                self._md5_id_map[md5] = uid
                if uid > max_uid:
                    max_uid = uid
                self._next_id = max_uid + 1
            else:
                raise RuntimeError("File in ingest failed to match expected "
                                   "naming format: '%s'" % filepath)

        if osp.isfile(self._eid_list_file):
            with self._eid_lock:
                with open(self._eid_list_file) as eidfile:
                    for line in eidfile.readlines():
                        self._eid_set.add(int(line.strip()))

    def _iter_ingest_files(self, d):
        """
        Iterates through files in the directory structure at the given directory

        :param d: Directory path
        :type d: str

        """
        for f in os.listdir(d):
            f = os.path.join(d, f)
            if os.path.isfile(f):
                yield f
            elif os.path.isdir(f):
                for e in self._iter_ingest_files(f):
                    yield e
            else:
                raise RuntimeError("Encountered something not a file or "
                                   "directory? :: '%s'" % f)

    def _get_next_id(self):
        """
        :return: The next data item ID to use
        :rtype: int
        """
        a = self._next_id
        self._next_id += 1
        return a

    @property
    def log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @property
    def data_directory(self):
        """
        :returns: Data directory for this ingest
        :rtype: str
        """
        if not os.path.isdir(self._data_dir):
            safe_create_dir(self._data_dir)
        return self._data_dir

    @property
    def work_directory(self):
        """
        :returns: Work directory for this ingest
        :rtype: str
        """
        if not os.path.isdir(self._work_dir):
            safe_create_dir(self._work_dir)
        return self._work_dir

    @property
    def content_directory(self):
        """
        :return: Sub-directory that contains the ingest data content tree
            (under the data directory).
        :rtype: str
        """
        d = osp.join(self.data_directory, "content")
        if not osp.isdir(d):
            safe_create_dir(d)
        return d

    def add_data_file(self, origin_filepath):
        """
        Add the given data file to this ingest

        The original file is copied and further maintenance of the original
        file is left to the user.

        If the given file exists in the ingest already, we do not add a second
        copy, instead returning the DataFile instance of the existing. Check max
        UID before and after this call to check for new ingest file or not.

        :param origin_filepath: Path to a file that should be added to this
            ingest.
        :type origin_filepath: str

        :return: The DataFile instance that was just ingested
        :rtype: DataFile

        """
        self.log.debug('Ingesting file: %s', origin_filepath)

        if not isinstance(origin_filepath, DataFile):
            origin_data = self.DATA_FILE_TYPE(origin_filepath)
        else:
            origin_data = origin_filepath

        # Overwriting last element in list so we don't have a single directory
        # per file. With 8 splits of the 16-element hex hash, ~65k files max per
        # leaf directory (16**16 total files).
        containing_dir = os.path.join(self.content_directory,
                                      *origin_data.split_md5sum(8)[:-1])
        # Copy original file into ingest
        md5 = origin_data.md5sum
        if md5 in self._md5_id_map:
            self.log.debug("File already ingested: %s -> %s",
                           origin_data.filepath, origin_data)
            return self._id_data_map[self._md5_id_map[md5]]

        if not os.path.isdir(containing_dir):
            os.makedirs(containing_dir)

        cur_id = self._get_next_id()
        origin_ext = os.path.splitext(origin_data.filepath)[1]

        fname = self.FILE_TEMPLATE % (cur_id, md5, origin_ext)
        target_filepath = os.path.join(containing_dir, fname)
        shutil.copy(origin_data.filepath, target_filepath)

        # Now that its copied over, we can make the data instance
        target_data = self.DATA_FILE_TYPE(target_filepath)
        target_data._uid = cur_id
        assert md5 == target_data.md5sum, \
            "Origin and target data files had divergent MD5 sums somehow!"

        self._id_data_map[cur_id] = target_data

        # if md5 not in self._md5_id_map:
        #     self._md5_id_map[md5] = set()
        # self._md5_id_map[md5].add(cur_id)
        self._md5_id_map[md5] = cur_id

        return target_data

    def iteritems(self):
        """
        Iterator over currently ingested data items.
        """
        return self._id_data_map.iteritems()

    def items(self):
        """
        :return: tuple of (UID, DataFile) pairs for data files ingested
        :rtype: tuple of (int, DataFile)
        """
        return self._id_data_map.items()

    def uids(self):
        """
        :return: list of the UIDs of data currently in this ingest.
        :rtype: list of int
        """
        return self._id_data_map.keys()

    def data_list(self):
        """
        :return: List of data elements in this ingest, sorted by UID
        :rtype: list of DataFile
        """
        return sorted(self._id_data_map.values(), key=lambda e: e.uid)

    def has_uid(self, uid):
        """
        Check if the given UID is in this ingest.

        :param uid: UID to check for
        :type uid: int

        :return: True of the given UID is in this ingest and false if not.
        :rtype: bool

        """
        return uid in self._id_data_map

    def get_data(self, uid):
        """
        :raises KeyError: The given UID does not exist in this ingest.

        :param uid: UID if the data item to get back
        :type uid: int

        :return: The ingested data element
        :rtype: DataFile

        """
        return self._id_data_map[uid]

    def get_uid(self, data_file):
        """
        If the given DataFile instance is in this ingest via MD5 sum, returning
        the UID of the last entry that matches MD5 sums, else returns None.

        :param data_file: DataFile instance to get the UID of.
        :type data_file: DataFile

        :return: UID of the DataFile instance or None if its not in this ingest.
        :rtype: int or None

        """
        if data_file.md5sum in self._md5_id_map:
            return sorted(self._md5_id_map[data_file.md5sum])[0]

    def max_uid(self):
        """
        :return: The highest value UID integer in this ingest. -1 if there is
            nothing in the ingest yet.
        :rtype: int
        """
        return max(self._id_data_map) if self._id_data_map else -1

    def is_explicit(self, uid):
        """
        Return whether the given file ID is marked as explicit.

        :raises KeyError: No element by the given UID in this ingest.

        :param uid: Unique ID of the item to check
        :type uid: int

        """
        with self._eid_lock:
            if uid in self._id_data_map:
                return uid in self._eid_set
            else:
                raise KeyError(uid)

    def set_explicit(self, uid):
        """
        Set the given file ID as explicit. This also updates this ingest's
        explicit ID list file if this ID wasn't labeled explicit before now.

        :raises KeyError: No element by the given UID in this ingest.

        :param uid: Item ID to set as explicit
        :type uid: int

        """
        with self._eid_lock:
            # only set and write out if this ID isn't already explicit
            if self.has_uid(uid) and uid not in self._eid_set:
                self._eid_set.add(uid)
                with open(self._eid_list_file, 'a') as elfile:
                    elfile.write('%d\n' % uid)
            else:
                raise KeyError(uid)


# TODO: Could probably add a ``compress`` function that creates a condensed
#       tar.gz of the ingest or something.

# TODO: Something to check for same-file ingest. Build map of MD5-to-idList,
#       check for associations lists > 1 in length.
