"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'paul.tunison'


import abc
import cPickle
import logging
import numpy as np
import pymongo
import time

from .utils import SafeConfigCommentParser
from .VCDStore import VCDStoreElement, VCDStore


__all__ = [
    'DistanceKernelInterface_Mongo'
]


class _timer (object):

    def __init__(self, msg):
        self._log = logging.getLogger('.'.join([self.__module__,
                                                self.__class__.__name__]))
        self._msg = msg
        self._s = 0.0

    def __enter__(self):
        self._log.info(self._msg)
        self._s = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.info("-> %f s", time.time() - self._s)


class DistanceKernelInterface (object):
    """
    Abstract interface for SVM distance kernel matrices. Normally these are
    square, but some implementations may contain rectangular matrices.
    These matrices are testing-only kernels, and will not implement the square
    sub-matrix method.

    These objects must be picklable.

    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def generate_config(cls, config=None):
        if config is None:
            config = SafeConfigCommentParser()

        return config

    @abc.abstractmethod
    def __init__(self, config):
        if config is None:
            config = self.generate_config()

        self._config = config

        self._log = logging.getLogger('.'.join([self.__module__,
                                                self.__class__.__name__]))

    def __getstate__(self):
        #noinspection PyRedundantParentheses
        return (self._config,)

    def __setstate__(self, state):
        self._log = logging.getLogger('.'.join([self.__module__,
                                                self.__class__.__name__]))
        self._config = state[0]

    @abc.abstractproperty
    def edge_clip_ids(self):
        """
        :return: The full matrix index-to-clipID tuple maps for the matrix.
            Returns 2 tuples of M (row) and N (column) length, respectively,
            when the matrix contained is of shape MxN.
        :rtype: (tuple of int, tuple of int)

        """
        return

    @abc.abstractmethod
    def get_sub_matrix(self, *clip_ids):
        """
        Return a symmetric sub NxN matrix of the total distance kernel based on
        the clip IDs provided. The background clips will always be included in
        the matrix.

        Clip IDs provided will be assumed non-background, or positive
        event examples. If the clip ID of a background video is provided as an
        argument, we will reconsider it as a non-background video for the
        returned data.

        :param clip_ids: Integer clip IDs to include in the returned matrix. The
            returned matrix will always contain all the Background videos.
        :type clip_ids: Iterable of int
        :return: Returns the index to clip ID map (tuple), the index to isBG map
            (tuple) and the symmetric NxN sub-matrix, where N is the number of
            clip IDs provided as arguments, plus the number of background
            videos.
        :rtype: (tuple of int, tuple of bool, numpy.ndarray)

        """
        return

    @abc.abstractmethod
    def extract_rows(self, *clipID_or_IDs):
        """
        Find and return the v-stacked distance vectors, in kernel row order
        (i.e. not the order given as arguments), of the kernel rows matching the
        given clip IDs.

        :param clipID_or_IDs: The integer clip ID or IDs of which to get the
            distances for.
        :type clipID_or_IDs: int or Iterable of int
        :return: The row index to clip ID map (tuple), the column index to clip
            ID map (tuple), and the KxL shape matrix, where K is the number of
            clip IDs give to this method, and L is the total width (columns) of
            the distance kernel.
        :rtype: (tuple of int, tuple of int, numpy.ndarray)

        """
        return


#class DistanceKernelInterface_Mongo (DistanceKernelInterface):
#    """
#    Interface class to the Database stored distance matrix, providing access as
#    if the matrix were stored locally.
#
#    Individual distances can be gotten using matrix access notation,
#    i.e. self[x, y] format.
#
#    """
#
#    DMI_COLLECTION = "DistanceMapData"
#
#    # _id of matrix row entries will be the integer index key of that row in the
#    #   constructed matrix
#    # A metadata construct will have the string ID with 3 elements with
#    #   following labels.
#    #   - metadata document format
#    DMI_MD_KEY = "DMI_METADATA"
#    DMI_DTYPE = "DMI_MATRIX_DTYPE"  # string
#    DMI_ORDERED_CID_TUP = "DMI_ORDERED_CID_TUP"  # picked tuple
#    DMI_ORDERED_ISBG_TUP = "DMI_ORDERED_BG_TUP"  # pickled tuple
#    # MongoDB Doc formats:
#    # -> Metadata:
#    #     { _id: <DMI_MD_KEY>,
#    #       <DMI_DTYPE>: <str>,
#    #       <DMI_IDX_CID_MAP>: <str>,
#    #       <DMI_IDX_ISBG_MAP>: <str> }
#    # -> Row Data:
#    #     { _id: <int_idx>,
#    #       row: <str> }
#
#    def __init__(self, db_info):
#        """
#        Construct the distance map interface.
#
#        :param db_info: Database connection information. The collection
#            attribute will be ignored as we specify our own.
#        :type db_info: DatabaseInfo
#
#        """
#        self._log = logging.getLogger('.'.join([self.__module__,
#                                                self.__class__.__name__]))
#
#        self._mdb_client = pymongo.MongoClient(db_info.host, db_info.port)
#        self._mdb_name = db_info.name
#
#        self._is_initialized = False
#        self._matrix_dtype = None  # stored, only needed when storing buffer
#        self._ordered_cid_tup = None  # stored
#        self._ordered_isBG_tup = None  # stored
#        self._cid_idx_map = None  # constructed
#        self._bg_cid_set = None  # constructed
#
#        # Attempt to load index maps. If either isn't there
#        coll = self._get_db_collection()
#        md_doc = coll.find_one({'_id': self.DMI_MD_KEY})
#        if md_doc:
#            self._log.info("Found existing data. Caching metadata.")
#            # If both are there we assume that there is existing matrix data
#            # loaded (initialize inputs matrix data before the metadata
#            # structures).s
#            self._is_initialized = True
#
#            # Retrieve stored values
#            self._matrix_dtype = md_doc[self.DMI_DTYPE]
#            self._ordered_cid_tup = cPickle.loads(str(md_doc[self.DMI_ORDERED_CID_TUP]))
#            self._ordered_isBG_tup = cPickle.loads(str(md_doc[self.DMI_ORDERED_ISBG_TUP]))
#
#            # Construct constructed values
#            self._cid_idx_map = dict((int(cid), idx)
#                                     for idx, cid
#                                     in enumerate(self._ordered_cid_tup))
#            self._bg_cid_set = frozenset(self._ordered_cid_tup[idx]
#                                         for idx, isBG
#                                         in enumerate(self._ordered_isBG_tup)
#                                         if isBG)
#
#    def _get_db_collection(self):
#        return self._mdb_client[self._mdb_name][self.DMI_COLLECTION]
#
#    def initialize_data(self, id_index_map, bg_data_map, npy_data_file):
#        """
#        Initialize the data stored in the database. If this structure was
#        already initialized, or if there is pre-existing data in the database,
#        we delete it and replace it with the provided data.
#
#        :param id_index_map: Path to the file listing the matrix index to clip
#            ID relationships.
#        :type id_index_map: str
#        :param bg_data_map: Path to the file listing flags for matrix indices
#            of whether the video for that index is considered a "background"
#            video.
#        :type bg_data_map: str
#        :param npy_data_file: Path to the numpy binary file containing the
#            matrix data.
#        :type npy_data_file: str
#
#        """
#        self._log.info("Initializing and loading IQR distance matrix")
#        coll = self._get_db_collection()
#
#        self._log.info("Dropping any existing data in DB")
#        coll.drop()
#
#        self._log.info("Loading ID to index map")
#        with open(id_index_map) as ifile:
#            self._ordered_cid_tup = tuple(int(cid) for cid in ifile.readlines())
#
#        self._cid_idx_map = dict((int(cid), idx)
#                                 for idx, cid in enumerate(self._ordered_cid_tup))
#
#        self._log.info("Loading index to isBD map")
#        with open(bg_data_map) as ifile:
#            self._ordered_isBG_tup = tuple(bool(int(isBG)) for isBG
#                                           in ifile.readlines())
#
#        self._bg_cid_set = frozenset(self._ordered_cid_tup[idx]
#                                     for idx, isBG
#                                     in enumerate(self._ordered_isBG_tup)
#                                     if isBG)
#
#        self._log.info("Loading IQR distance matrix")
#        s = time.time()
#        mat = np.load(npy_data_file)
#        self._matrix_dtype = str(mat.dtype)
#        self._log.info("-> Time to load: %f s", time.time() - s)
#
#        self._log.info("creating matrix database documents")
#        to_insert = []
#        push_trigger = 1024 * 8
#        for idx, row in enumerate(mat):
#            if idx % 100 == 0:
#                self._log.info("Num packaged: %d", idx)
#
#            doc = {
#                '_id': idx,
#                'row': cPickle.dumps(row)
#                # Raises some invalid string error in mongo
#                #'row': str(buffer(row)),
#            }
#            to_insert.append(doc)
#
#            if len(to_insert) >= push_trigger:
#                self._log.info("Inserting batch into DB (n: %d)", push_trigger)
#                s = time.time()
#                coll.insert(to_insert)
#                self._log.info("-> Total time for matrix insertion: %f s",
#                               time.time() - s)
#                to_insert = []
#
#        self._log.info("Inserting FINAL batch into DB (n: %d)", len(to_insert))
#        s = time.time()
#        coll.insert(to_insert)
#        self._log.info("-> Total time for matrix insertion: %f s",
#                       time.time() - s)
#
#        self._log.info('Inserting and caching metadata')
#        s = time.time()
#        md_doc = {
#            '_id': self.DMI_MD_KEY,
#            self.DMI_DTYPE: self._matrix_dtype,
#            self.DMI_ORDERED_CID_TUP: cPickle.dumps(self._ordered_cid_tup),
#            self.DMI_ORDERED_ISBG_TUP: cPickle.dumps(self._ordered_isBG_tup)
#        }
#        coll.insert(md_doc)
#        self._log.info("Metadata insertion time: %f s", time.time() - s)
#
#        # NOT adding an index on _id, as it exists by default.
#
#        # queue matrix clean-up from RAM
#        del mat
#
#        self._is_initialized = True
#
#    @property
#    def is_initialized(self):
#        return self._is_initialized
#
#    def get_sub_matrix(self, *clip_ids):
#        """
#        Return a sub NxN matrix of the total distance map based on the video IDs
#        provided. The background clips will always be included in the matrix.
#
#        :param clip_ids: Integer clip IDs to include in the returned matrix. The
#            returned matrix will always contain all the Background videos.
#        :type clip_ids: Iterable of int
#        :return: Returns the clip ID to index map (dict), the index to isBG map
#            (tuple) and the NxN sub-matrix, where N is the number of clip IDs
#            provided as arguments, plus the number of background videos.
#        :rtype: (dict of (int, int), tuple of bool, numpy.ndarray)
#
#        """
#        self._log.info("Starting sub-matrix retrieval and extraction")
#        t_s = time.time()
#        if not self._is_initialized:
#            raise RuntimeError("No data initialized yet!")
#
#        assert all((isinstance(e, int) for e in clip_ids)), \
#            "Not all clip IDs given were integers! This is required."
#
#        # Matrix to return should be the distance matrix of all background clips
#        # as well as clips provided
#        # - Determine what clips IDs are background clips
#        # - create set of clip IDs that are the union of background clips and
#        #   those provided
#        # - iteratively fetch rows from DB for clip ID set, extracting pertinent
#        #   columns, v-stacking row-ified columns to produce results symmetric
#        #   matrix.
#
#        all_cids = self._bg_cid_set.union(clip_ids)
#
#        # Create a list of clip IDs that are in the same relative order as the
#        # total set
#        self._log.info("Creating focus index sequence from master sequence.")
#        s = time.time()
#        focus_indices = []
#        for idx, cid in enumerate(self._ordered_cid_tup):
#            if cid in all_cids:
#                focus_indices.append(idx)
#        self._log.info("-> %f s", time.time() - s)
#
#        N = len(focus_indices)
#        focus_cid2idx = {}
#        focus_id2isBG = []
#        coll = self._get_db_collection()
#
#        self._log.info("Creating metadata structures")
#        s = time.time()
#        for new_idx, idx in enumerate(focus_indices):
#            focus_cid2idx[self._ordered_cid_tup[idx]] = new_idx
#            focus_id2isBG.append(self._ordered_isBG_tup[idx])
#        self._log.info("-> %f s", time.time() - s)
#
#        #######################################################################
#        ### single-element construction method
#        self._log.info("Creating sub-matrix")
#        s = time.time()
#        ret_mat = np.zeros((N, N), dtype=self._matrix_dtype)
#        for i, idx in enumerate(focus_indices):
#            doc = coll.find_one({'_id': idx})
#            assert doc, "Missing matrix row entry for index %d (cid:%d)" \
#                        % (idx, self._ordered_cid_tup[idx])
#            row = cPickle.loads(str(doc['row']))
#            for j, _idx in enumerate(focus_indices):
#                ret_mat[i, j] = row[_idx]
#        self._log.info("-> %f s", time.time() - s)
#
#        #######################################################################
#        ### row->column extraction method
#        #self._log.info("Creating metadata structures")
#        #s = time.time()
#        #for new_idx, idx in enumerate(focus_indices):
#        #    focus_cid2idx[self._ordered_cid_tup[idx]] = new_idx
#        #    focus_id2isBG.append(self._ordered_isBG_tup[idx])
#        #self._log.info("-> %f s", time.time() - s)
#        #
#        #self._log.info("Collecting rows")
#        #s = time.time()
#        #pickled_rows = []
#        #for idx in focus_indices:
#        #    pickled_rows.append(coll.find_one({'_id': idx})['row'])
#        ##ret = coll.find({'_id': {"$in": focus_indices}})
#        #self._log.info('-> %f s', time.time() - s)
#        #
#        #self._log.info("Un-pickling data")
#        #s = time.time()
#        #rows = []
#        #p_r = None
#        #for p_r in pickled_rows:
#        #    rows.append(cPickle.loads(str(p_r)))
#        #del pickled_rows, p_r
#        #self._log.info("-> %f s", time.time() - s)
#        #
#        #self._log.info("Consolidating into wide matrix")
#        #s = time.time()
#        #wide_mat = np.mat(rows)
#        #self._log.info("-> %f s", time.time() - s)
#        #
#        #self._log.info("Extracting columns from wide matrix")
#        #s = time.time()
#        #cols = []
#        #for idx in focus_indices:
#        #    cols.append(wide_mat[:, idx])
#        #self._log.info("-> %f s", time.time() - s)
#        #
#        #self._log.info("Constructing final matrix")
#        #s = time.time()
#        ## Because of the symmetric nature of the extracted sub-matrix, and since
#        ## numpy row-ifies the columns when extracted
#        #ret_mat = np.hstack(cols)
#        #self._log.info("-> %f s", time.time() - s)
#
#        self._log.info("==> Total: %f s", time.time() - t_s)
#        return focus_cid2idx, focus_id2isBG, ret_mat
#
#    def get_clip_distances(self, clip_id):
#        """
#        Find and return the vector of distance of this clip to all other clips.
#
#        :param clip_id: The integer clip ID of which to get the distances for.
#        :type clip_id: int
#        :return: The clip ID to index map (dict), the index to isBG map
#            (tuple), and the N length vector, where N is the total number of
#            clip IDs in the distance map.
#        :rtype: (dict of (int, int), tuple of bool, numpy.ndarray)
#
#        """
#        if not self._is_initialized:
#            raise RuntimeError("No data initialized yet!")
#
#        coll = self._get_db_collection()
#        doc = coll.find_one({'_id': self._cid_idx_map[clip_id]})
#        row = cPickle.loads(str(doc['row']))
#        return self._cid_idx_map, self._ordered_isBG_tup, row
#
#    def __getitem__(self, id_1, id_2):
#        """
#        Access a single distance from the matrix.
#
#        :raises KeyError: If one or both IDs are not included in the distance
#            map.
#
#        :param id_1: An integer clip id
#        :type id_1: int
#        :param id_2: An integer clip id
#        :type id_2: int
#        :return: Distance between the specified videos.
#        :rtype: float
#
#        """
#        if not self._is_initialized:
#            raise RuntimeError("No data initialized yet!")
#
#        assert isinstance(id_1, int) and isinstance(id_2, int), \
#            "require integer clip IDs!"
#
#        row_idx = self._cid_idx_map[id_1]
#        col_idx = self._cid_idx_map[id_2]
#
#        raise NotImplementedError()
#
#
#class DistanceKernelInterface_SQLite3 (DistanceKernelInterface):
#    """
#    Distance map shared matrix implementation using SQLite3
#
#
#    Table Specification (when we get to using our own sql impl)
#    ===========================================================
#
#    MAT_DATA
#    --------
#    idx, INTEGER NOT NULL
#    row, BLOB NOT NULL
#
#    METADATA
#    --------
#    label, TEXT
#    value, TEXT
#
#    """
#
#    DMI_COLLECTION = "DistanceMapDataSQL"
#
#    # A metadata construct will have the string ID with 3 elements with
#    #   following labels.
#    #   - metadata document format
#    DMI_MD_KEY = "DMI_METADATA"
#    DMI_DTYPE = "DMI_MATRIX_DTYPE"  # string
#    DMI_ORDERED_CID_TUP = "DMI_ORDERED_CID_TUP"  # picked tuple
#    DMI_ORDERED_ISBG_TUP = "DMI_ORDERED_BG_TUP"  # pickled tuple
#    # MongoDB Doc format:
#    # -> Metadata:
#    #     { _id: <DMI_MD_KEY>,
#    #       <DMI_DTYPE>: <str>,
#    #       <DMI_IDX_CID_MAP>: <str>,
#    #       <DMI_IDX_ISBG_MAP>: <str> }
#
#    @classmethod
#    def generate_config(cls, config=None):
#        if config is None:
#            config = SafeConfigCommentParser()
#
#        #sect = "distance_map_interface"
#        #config.add_section(sect,
#        #                   "This interface inherits database connection "
#        #                   "information from the instantiating agent.")
#
#        return config
#
#    def __init__(self, db_path, mongo_dbinfo):
#        """
#        Initialize DMI to use specific database location
#
#        :param db_path: Path to the database file
#        :type db_path: str
#        :param mongo_dbinfo: Database information to mongo DB to connect to for
#            metadata storage.
#        :type mongo_dbinfo: DatabaseInfo
#
#        """
#        self._log = logging.getLogger('.'.join([self.__module__,
#                                                self.__class__.__name__]))
#
#        self._mat_vcds = VCDStore(fs_db_path=db_path)
#        self._mdb_client = pymongo.MongoClient(mongo_dbinfo.host,
#                                               mongo_dbinfo.port)
#        self._mdb_name = mongo_dbinfo.name
#
#        self._is_initialized = False
#        self._matrix_dtype = None  # stored, only needed when storing buffer
#        self._ordered_cid_tup = None  # stored
#        self._ordered_isBG_tup = None  # stored
#        self._cid_idx_map = None  # constructed
#        self._bg_cid_set = None  # constructed
#
#        # Check if database is initialize by looking for table and metadata
#        # existence.
#        coll = self._get_mdb_collection()
#        md_doc = coll.find_one({'_id': self.DMI_MD_KEY})
#        if md_doc:
#            self._log.info("Found existing data. Caching metadata.")
#            s = time.time()
#            # If both are there we assume that there is existing matrix data
#            # loaded (initialize inputs matrix data before the metadata
#            # structures).s
#            self._is_initialized = True
#
#            # Retrieve stored values
#            self._matrix_dtype = md_doc[self.DMI_DTYPE]
#            self._ordered_cid_tup = cPickle.loads(str(md_doc[self.DMI_ORDERED_CID_TUP]))
#            self._ordered_isBG_tup = cPickle.loads(str(md_doc[self.DMI_ORDERED_ISBG_TUP]))
#
#            # Construct constructed values
#            self._cid_idx_map = dict((int(cid), idx)
#                                     for idx, cid
#                                     in enumerate(self._ordered_cid_tup))
#            self._bg_cid_set = frozenset(self._ordered_cid_tup[idx]
#                                         for idx, isBG
#                                         in enumerate(self._ordered_isBG_tup)
#                                         if isBG)
#            self._log.info("-> %f s", time.time() - s)
#
#    def _get_mdb_collection(self):
#        return self._mdb_client[self._mdb_name][self.DMI_COLLECTION]
#
#    @property
#    def is_initialized(self):
#        return self._is_initialized
#
#    def initialize_data(self, id_index_map, bg_data_map, npy_data_file):
#        """
#        Initialize the data stored in the database. If this structure was
#        already initialized, or if there is pre-existing data in the database,
#        we delete it and replace it with the provided data.
#
#        :param id_index_map: Path to the file listing the matrix index to clip
#            ID relationships.
#        :type id_index_map: str
#        :param bg_data_map: Path to the file listing flags for matrix indices
#            of whether the video for that index is considered a "background"
#            video.
#        :type bg_data_map: str
#        :param npy_data_file: Path to the numpy binary file containing the
#            matrix data.
#        :type npy_data_file: str
#
#        """
#        self._log.info("Initializing and loading IQR distance matrix")
#
#        self._log.info("Dropping any existing data in DB")
#
#        self._log.info("Loading ID to index map")
#        with open(id_index_map) as ifile:
#            self._ordered_cid_tup = tuple(int(cid) for cid in ifile.readlines())
#
#        self._cid_idx_map = dict((int(cid), idx)
#                                 for idx, cid
#                                 in enumerate(self._ordered_cid_tup))
#
#        self._log.info("Loading index to isBD map")
#        with open(bg_data_map) as ifile:
#            self._ordered_isBG_tup = tuple(bool(int(isBG))
#                                           for isBG in ifile.readlines())
#
#        self._bg_cid_set = frozenset(self._ordered_cid_tup[idx]
#                                     for idx, isBG
#                                     in enumerate(self._ordered_isBG_tup)
#                                     if isBG)
#
#        self._log.info("Loading IQR distance matrix")
#        s = time.time()
#        mat = np.load(npy_data_file)
#        self._matrix_dtype = str(mat.dtype)
#        self._log.info("-> %f s", time.time() - s)
#
#        self._log.info("Inserting matrix data into SQLite3 database")
#
#        self._log.info("Creating store elements")
#        elements = []
#        for idx, row in enumerate(mat):
#            e = VCDStoreElement(self.DMI_COLLECTION, idx, row)
#            elements.append(e)
#
#        self._log.info("Inserting elements into database")
#        s = time.time()
#        self._mat_vcds.store_feature(elements)
#        self._log.info("Time to insert: %f s", time.time() - s)
#
#    #def get_sub_matrix(self, *clip_ids):


class DistanceKernel_File_IQR (DistanceKernelInterface):
    """
    Load and perform functions on a symmetric distance matrix from file intended
    for IQR learning and searching.
    """

    def __init__(self, id_index_map, bg_data_map, npy_data_file):
        """
        :param id_index_map: Path to the file listing the matrix index to clip
            ID relationships.
        :type id_index_map: str
        :param bg_data_map: Path to the file listing flags for matrix indices
            of whether the video for that index is considered a "background"
            video.
        :type bg_data_map: str
        :param npy_data_file: Path to the numpy binary file containing the
            matrix data.
        :type npy_data_file: str

        """
        super(DistanceKernel_File_IQR, self).__init__(None)

        self._npy_data_file = npy_data_file

        with open(id_index_map) as ifile:
            self._ordered_cid_tup = tuple(int(cid) for cid in ifile.readlines())

        with open(bg_data_map) as ifile:
            self._ordered_isBG_tup = tuple(bool(int(isBG))
                                           for isBG in ifile.readlines())

        self._bg_cid_set = frozenset(self._ordered_cid_tup[idx]
                                     for idx, isBG
                                     in enumerate(self._ordered_isBG_tup)
                                     if isBG)

    def __getstate__(self):
        return (
            super(DistanceKernel_File_IQR, self).__getstate__(),
            self._npy_data_file,
            self._ordered_cid_tup,
            self._ordered_isBG_tup,
            self._bg_cid_set
        )

    def __setstate__(self, state):
        super(DistanceKernel_File_IQR, self).__setstate__(state[0])
        self._npy_data_file = state[1]
        self._ordered_cid_tup = state[2]
        self._ordered_isBG_tup = state[3]
        self._bg_cid_set = state[4]

    @property
    def edge_clip_ids(self):
        return self._ordered_cid_tup, self._ordered_cid_tup

    def get_sub_matrix(self, *clip_ids):
        """
        Return a symmetric sub NxN matrix of the total distance kernel based on
        the clip IDs provided. The background clips will always be included in
        the matrix.

        Clip IDs provided will be assumed non-background, or positive
        event examples. If the clip ID of a background video is provided as an
        argument, we will reconsider it as a non-background video for the
        returned data.

        :param clip_ids: Integer clip IDs to include in the returned matrix. The
            returned matrix will always contain all the Background videos.
        :type clip_ids: Iterable of int
        :return: Returns the index to clip ID map (tuple), the index to isBG map
            (tuple) and the symmetric NxN sub-matrix, where N is the number of
            clip IDs provided as arguments, plus the number of background
            videos.
        :rtype: (tuple of int, tuple of bool, numpy.ndarray)

        """
        self._log.info("Starting symmetric sub-matrix retrieval and extraction")
        s_t = time.time()

        assert all((isinstance(e, int) for e in clip_ids)), \
            "Not all clip IDs given were integers! This is required."
        assert not set(clip_ids).difference(self._ordered_cid_tup), \
            "Not all clip IDs provided are represented in the distance " \
            "kernel matrix row map! (difference: %s)" \
            % set(clip_ids).difference(self._ordered_cid_tup)

        all_cids = self._bg_cid_set.union(clip_ids)

        # Create a list of clip IDs that are in the same relative order as the
        # total set. If there are duplicates, we only pick the first one.
        with _timer("Creating focus index sequence from master sequence"):
            focus_indices = []
            focus_clipIDs = []
            for idx, cid in enumerate(self._ordered_cid_tup):
                # Filtering out IDs that we have already seen
                if cid in all_cids and (cid not in focus_clipIDs):
                    focus_indices.append(idx)
                    focus_clipIDs.append(cid)

        focus_id2isBG = []

        with _timer("Creating metadata structures"):
            for idx in focus_indices:
                cid = self._ordered_cid_tup[idx]

                # IDs provided as arguments are considered non-background, while
                # all other are considered background (those added with the set
                # union above)
                focus_id2isBG.append(False if cid in clip_ids else True)

        with _timer("Loading full kernel matrix"):
            full_kernel_mat = np.load(self._npy_data_file)

        with _timer("Creating sub-matrix"):
            # Apparently this is just some special super efficient syntax for np
            ret_mat = full_kernel_mat[focus_indices, :][:, focus_indices]

        self._log.info("==> Total: %f s", time.time() - s_t)
        return focus_clipIDs, focus_id2isBG, ret_mat

    def extract_rows(self, *clipID_or_IDs):
        """
        Find and return the v-stacked distance vectors, in kernel row order
        (i.e. not the order given as arguments), of the kernel rows matching the
        given clip IDs.

        :param clipID_or_IDs: The integer clip ID or IDs of which to get the
            distances for.
        :type clipID_or_IDs: int or Iterable of int
        :return: The row index to clip ID map (tuple), the column index to clip
            ID map (tuple), and the KxL shape matrix, where K is the number of
            clip IDs give to this method, and L is the total width (columns) of
            the distance kernel.
        :rtype: (tuple of int, tuple of int, numpy.ndarray)

        """
        self._log.info("Starting kernel row retrieval and extraction")
        s_t = time.time()

        assert all((isinstance(e, int) for e in clipID_or_IDs)), \
            "Not all clip IDs given were integers! This is required."
        assert not set(clipID_or_IDs).difference(self._ordered_cid_tup), \
            "Not all clip IDs provided are represented in the distance " \
            "kernel matrix row map! (difference: %s)" \
            % set(clipID_or_IDs).difference(self._ordered_cid_tup)

        with _timer("Loading full kernel matrix"):
            full_kernel_mat = np.load(self._npy_data_file)

        # Create ordered tuple of clip IDs that are in the same relative order
        # as the kernel matrix's edge order.
        with _timer("Creating focus index/cid sequence"):
            focus_indices = []
            focus_clipIDs = []
            for idx, cid in enumerate(self._ordered_cid_tup):
                # Filtering out IDs that we have already seen
                if cid in clipID_or_IDs and (cid not in focus_clipIDs):
                    focus_indices.append(idx)
                    focus_clipIDs.append(cid)

        with _timer("Cropping kernel to focus indices"):
            wide_mat = full_kernel_mat[focus_indices, :]

        self._log.info("==> Total: %f s", time.time() - s_t)
        return focus_clipIDs, self._ordered_cid_tup, wide_mat


class DistanceKernel_File_Archive (DistanceKernelInterface):
    """
    Load and perform functions on a distance matrix from file intended for IQR
    learning and searching. There is no guarantee that this matrix will be
    symmetric or that clip IDs will be at all shared along axes.
    """

    def __init__(self, id_index_map_rows, id_index_map_cols, npy_data_file):
        """
        :param id_index_map_rows: Path to the file detailing the index-to-clipID
            relationship of the rows of the matrix.
        :type id_index_map_rows: str
        :param id_index_map_cols: Path to the file detailing the index-to-clipID
            relationship of the columns of the matrix.
        :type id_index_map_cols: str
        :param npy_data_file: Path to the numpy binary file containing the
            matrix data.
        :type npy_data_file: str

        """
        super(DistanceKernel_File_Archive, self).__init__(None)

        self._npy_data_file = npy_data_file

        with open(id_index_map_rows) as ifile:
            self._ordered_row_cid_tup = tuple(int(cid) for cid
                                              in ifile.readlines())

        with open(id_index_map_cols) as ifile:
            self._ordered_col_cid_tup = tuple(int(cid) for cid
                                              in ifile.readlines())

    def __getstate__(self):
        return (
            super(DistanceKernel_File_Archive, self).__getstate__(),
            self._npy_data_file,
            self._ordered_row_cid_tup,
            self._ordered_col_cid_tup
        )

    def __setstate__(self, state):
        super(DistanceKernel_File_Archive, self).__setstate__(state[0])
        self._npy_data_file = state[1]
        self._ordered_row_cid_tup = state[2]
        self._ordered_col_cid_tup = state[3]

    @property
    def edge_clip_ids(self):
        return self._ordered_row_cid_tup, self._ordered_col_cid_tup

    def get_sub_matrix(self, *clip_ids):
        """
        Undefined functionality for non-symmetric matrix
        """
        raise NotImplementedError("Symmetric matrix sub-duvision undefined "
                                  "for non-symmetric matrices")

    def extract_rows(self, *clipID_or_IDs):
        """
        Find and return the v-stacked distance vectors, in kernel row order
        (i.e. not the order given as arguments), of the kernel rows matching the
        given clip IDs.

        :param clipID_or_IDs: The integer clip ID or IDs of which to get the
            distances for.
        :type clipID_or_IDs: int or Iterable of int
        :return: The row index to clip ID map (tuple), the column index to clip
            ID map (tuple), and the KxL shape matrix, where K is the number of
            clip IDs give to this method, and L is the total width (columns) of
            the distance kernel.
        :rtype: (tuple of int, tuple of int, numpy.ndarray)

        """
        self._log.info("Starting kernel row retrieval and extraction")
        s_t = time.time()

        assert all((isinstance(e, int) for e in clipID_or_IDs)), \
            "Not all clip IDs given were integers! This is required."
        assert not set(clipID_or_IDs).difference(self._ordered_row_cid_tup), \
            "Not all clip IDs provided are represented in the distance " \
            "kernel matrix row map! (difference: %s)" \
            % set(clipID_or_IDs).difference(self._ordered_row_cid_tup)

        with _timer("Loading full kernel matrix"):
            full_kernel_mat = np.load(self._npy_data_file)

        # Create ordered tuple of clip IDs that are in the same relative order
        # as the kernel matrix's edge order.
        with _timer("Creating focus index/cid sequences"):
            focus_row_indices = []
            focus_row_clipIDs = []
            for idx, cid in enumerate(self._ordered_row_cid_tup):
                # Filtering out IDs that we have already seen
                if cid in clipID_or_IDs and (cid not in focus_row_clipIDs):
                    focus_row_indices.append(idx)
                    focus_row_clipIDs.append(cid)

        with _timer("Cropping kernel to focus indices"):
            wide_mat = full_kernel_mat[focus_row_indices, :]

        self._log.info("==> Total: %f s", time.time() - s_t)
        return focus_row_clipIDs, self._ordered_col_cid_tup, wide_mat
