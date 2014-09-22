"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Specific backend implementation that leveraged an SQLite3 database to store
features.

"""

import numpy
import os
import os.path as osp
import sqlite3

from . import VCDStoreBackendInterface
from ..VCDStoreElement import VCDStoreElement
from ..errors import *


class SQLiteVCDStoreBackend(VCDStoreBackendInterface):
    """
    A database specific implementation of the VCDStore using sqlite3.

    For ``None`` fields in VCDStoreElement object, stores a -1 in the database
    since there should never be a -1 value for any of the metadata fields
    anyway.

    """

    # The default path where store files/databases will be recorded. This may be
    # changed before the construction of a FrameStore object to change where the
    # object will look for possibly existing files/databases.
    DEFAULT_DB_NAME = 'SQLiteVCDStore.db'

    # Default table name
    DEFAULT_TABLE_NAME = 'feature_store'

    # Column details (name, sql_type + constraints)
    __COL_DEF = (
        ('descriptor_id', 'TEXT NOT NULL'),
        ('video_id', 'INTEGER NOT NULL'),
        ('frame_num', 'INTEGER NOT NULL'),
        ('timestamp', 'REAL NOT NULL'),
        ('spacial_x', 'REAL NOT NULL'),
        ('spacial_y', 'REAL NOT NULL'),
        ('feature_vec', 'BLOB NOT NULL')
    )

    # Insert command update.
    # This should take:
    #   - fallback: the conflict mode
    #   - table_name: the table name to insert into
    #   - col_list: a string of the comma-separated column names for the table
    #   - placeholder_list: a string of comma-separated place holder string
    #                       values ('?'0). This needs to be exactly as long as
    #                       the column list.
    __INSERT__TMPL = ("INSERT OR {fallback} INTO {table_name} "
                      "({col_list}) VALUES ({placeholder_list});")

    # Select cmd template. Returns the feature vector blob for each row
    # selected.
    # This should take:
    #   - col_name: The name of the column that we want to select (or multiple
    #               column names, separated by commas
    #   - table_name: the name of the table to select from
    #   - where_criteria: A string of comma-separated column constraints for the
    #                     select.
    __SELECT_TMPL = "SELECT {cols} FROM {table_name}"

    def __init__(self, fs_db_path=None, db_root=None):
        """
        Sqlite specific FrameStoreBackend implementation. If explicit values are
        not provided for the feature database locations, defaults are used..

        :param fs_db_path: The path to the feature store database file. If not
            provided, this defaults to "{db_root}/{DEFAULT_DB_NAME}". This may
            also be an absolute path, which would cause the db_root to not be
            used, If a relative path is given it is interpreted relative to the
            given ``db_root`` (which defaults to the current working directory).
        :type fs_db_path: str
        :param db_root: The root directory to create / look for existing
            databases. If this database is not provided, we will use the current
            working directory.
        :type db_root: str

        """
        # immediate super method does nothing.
        super(SQLiteVCDStoreBackend, self).__init__()

        ###
        # Resolve database locations
        #
        self._db_root = osp.abspath(osp.expanduser(db_root)) if db_root \
            else os.getcwd()

        self._db_path = \
            osp.join(self._db_root, osp.expanduser(fs_db_path)) if fs_db_path \
            else osp.join(self._db_root, self.DEFAULT_DB_NAME)

        # reassign db_root again as fs_db_path may have been an absolute path
        self._db_root = osp.dirname(self._db_path)

        # Check that database directory exists. If not, create it.
        if not osp.isdir(self._db_root):
            os.makedirs(self._db_root)

        ###
        # Initialize database connections.
        #
        # Also attempt to make the table for each database if it doesn't exist
        #
        self._db_conn = sqlite3.connect(self._db_path)
        self._db_cursor = self._db_conn.cursor()
        self._db_cursor.execute(self.__create_table_cmd())
        self._db_conn.commit()

    def __del__(self):
        self._db_cursor.close()
        self._db_conn.close()

    def __create_table_cmd(self):
        """ Construct and return the create table SQL command.

        This uses the __COL_DEF structure to generated the command. Every column
        except the last will be treated as primary keys. The last column should
        always be the feature vector BLOB column.

        :return: The table creation command as a string.
        :rtype: str

        """
        cmd_template = ("CREATE TABLE IF NOT EXISTS %(table_name)s "
                        "("
                        " %(col_defs)s,"
                        " PRIMARY KEY (%(pk_list)s)"
                        ");")

        # construct PK col listing from everything but last column
        pk_cols = [n for n, _ in self.__COL_DEF[:-1]]
        pk_cols = ', '.join(pk_cols)

        # construct the column definition body
        col_defs_body = ', '.join(['%s %s' % (n, t) for n, t in self.__COL_DEF])

        # flesh out template and return
        return cmd_template % {'table_name': self.DEFAULT_TABLE_NAME,
                               'col_defs': col_defs_body,
                               'pk_list': pk_cols}

    def __gen_insert_cmd(self, table_name, col_list, overwrite=False):
        """
        Generate a generic insert command for the given column list with the
        standard sqlite placeholder characters for use with the sqlite execute
        command. We can optionally set the call to overwrite existing values.

        :param table_name: The name of the table to insert into.
        :type table_name: str
        :param col_list: The list of column names to insert into.
        :type col_list: list of str or tuple of str
        :param overwrite: Allow the command to overwrite an existing value.
        :type overwrite: bool
        :return: The constructed INSERT command to be used in the execute method
            of a database cursor.
        :rtype: str

        """
        fallback = 'REPLACE' if overwrite else 'ABORT'

        # Placeholder list ('?') must be the same length as the column list
        ph_list = ', '.join(['?'] * len(col_list))
        column_list = ', '.join(col_list)

        return self.__INSERT__TMPL.format(fallback=fallback,
                                          table_name=table_name,
                                          col_list=column_list,
                                          placeholder_list=ph_list)

    def __gen_select_cmd(self, table_name, select_col_list,
                         where_col_list=None):
        """
        Generate a generic select command for the sqlite execute command.
        Optionally include one or more where clause qualification placeholders.

        :param table_name: The name of the table to select from.
        :type table_name: str
        :param select_col_list: The column names for select out in the query.
        :type select_col_list: tuple of str or list of str
        :param where_col_list: The columns to create placeholders for in the
            WHERE clause.
        :type where_col_list: tuple of str or list of str
        :return: The constructed SELECT command to be used in the execute method
            of a database cursor.
        :rtype: str

        """
        select_cols = ', '.join(select_col_list)
        stmt = self.__SELECT_TMPL.format(cols=select_cols,
                                         table_name=table_name)

        if where_col_list:
            where_criteria = ' WHERE ' + ' AND '.join(['%s IS ?' % c
                                                       for c in where_col_list])
            stmt += where_criteria

        return stmt

    def __generic_store(self, cursor, db, cmd, value_tuple):
        """
        :type cursor: sqlite3.Cursor
        :type db: sqlite3.Connection
        :type cmd: str
        :type value_tuple: tuple or list
        """
        try:
            cursor.executemany(cmd, value_tuple)
            db.commit()
        except sqlite3.IntegrityError as ex:
            self._log.warn("Integrity Error. Rolling back. (error: %s)",
                           str(ex))
            db.rollback()
            raise VCDDuplicateFeatureError("Possible duplicate entry for keys: %s"
                                        % str(value_tuple))

    def __generic_get(self, cursor, cmd, value_tuple):
        """ Returns a tuple of the returned rows.

        This will always be at least a tuple of one element.

        :type cursor: sqlite3.Cursor
        :type cmd: str
        :type value_tuple: tuple or list
        :rtype: numpy.ndarray
        """
        cursor.execute(cmd, value_tuple)
        r_list = cursor.fetchall()

        return r_list

    def store_feature(self, feature_elements, overwrite=False):
        # Store one or more VCDStoreElement entries. Unless ``feature_elements``
        # is a VCDStoreElement object, assume its iterable.
        if isinstance(feature_elements, VCDStoreElement):
            feature_elements = (feature_elements,)

        col_list = [e[0] for e in self.__COL_DEF]
        cmd = self.__gen_insert_cmd(self.DEFAULT_TABLE_NAME,
                                    col_list=col_list,
                                    overwrite=overwrite)

        # transform feature vector list into a list of buffers for each numpy
        # ndarray
        rows = list()
        for fs in feature_elements:
            # same order as __COL_DEF structure
            rows.append((fs.descriptor_id,
                         fs.video_id,
                         fs.frame_num if fs.frame_num is not None else -1,
                         fs.timestamp if fs.timestamp is not None else -1,
                         fs.spacial_x if fs.spacial_x is not None else -1,
                         fs.spacial_y if fs.spacial_y is not None else -1,
                         buffer(fs.feat_vec)))

        self.__generic_store(self._db_cursor, self._db_conn, cmd, rows)

    def get_feature(self, descriptor_id, video_id, frame_num=None,
                    timestamp=None, spacial_x=None, spacial_y=None):

        select_cols = [self.__COL_DEF[-1][0]]  # just want the feature vector
                                               # for this get function

        # where clause and values will always contain all keys so as to only
        # find specifically matching features.
        where_cols = [e[0] for e in self.__COL_DEF[:-1]]
        values = (descriptor_id, video_id,
                  frame_num if frame_num >= 0 else -1,
                  timestamp if timestamp >= 0 else -1,
                  spacial_x if spacial_x >= 0 else -1,
                  spacial_y if spacial_y >= 0 else -1)

        cmd = self.__gen_select_cmd(self.DEFAULT_TABLE_NAME,
                                    select_cols,
                                    where_cols)
        # print "Generated command:", cmd
        # print "Input values:", values
        # print "Raw return:", self.__generic_get(self._db_cursor, cmd, values)
        ret_vals = self.__generic_get(self._db_cursor, cmd, values)

        if not ret_vals:  # no returned results for query
            raise VCDNoFeatureError("No feature for the given query")

        np_buffer = ret_vals[0][0]
        #: :type: numpy.ndarray
        feat_vec = numpy.frombuffer(np_buffer)

        return VCDStoreElement(descriptor_id, video_id, feat_vec,
                              frame_num if frame_num != -1 else None,
                              timestamp if frame_num != -1 else None,
                              spacial_x if frame_num != -1 else None,
                              spacial_y if frame_num != -1 else None)

    def get_features_by(self, descriptor_id=None, video_id=None, frame_num=None,
                        timestamp=None, spacial_x=None, spacial_y=None):
        select_cols = [e[0] for e in self.__COL_DEF]

        # for each key, add a where clause and add to the value list
        where_cols = []
        values = []
        if descriptor_id is not None:
            where_cols += [self.__COL_DEF[0][0]]
            values += [descriptor_id]
        if video_id is not None:
            where_cols += [self.__COL_DEF[1][0]]
            values += [video_id]
        if frame_num is not None:
            where_cols += [self.__COL_DEF[2][0]]
            values += [frame_num]
        if timestamp is not None:
            where_cols += [self.__COL_DEF[3][0]]
            values += [timestamp]
        if spacial_x is not None:
            where_cols += [self.__COL_DEF[4][0]]
            values += [spacial_x]
        if spacial_y is not None:
            where_cols += [self.__COL_DEF[5][0]]
            values += [spacial_y]

        cmd = self.__gen_select_cmd(self.DEFAULT_TABLE_NAME,
                                    select_cols,
                                    where_cols)

        raw = self.__generic_get(self._db_cursor, cmd, values)

        # formulate return content
        ret = []
        for t in raw:
            r = list(t)
            # - Transform any element that's '-1' into a None.
            for i, e in enumerate(r):
                if e == -1:
                    r[i] = None
            # - Transform feature vector in each returned tuple from a buffer
            # object back into a numpy.ndarray (since select cols is set to all
            # known columns).
            r[6] = numpy.frombuffer(r[6])
            # Create feature element knowing that r is in the same order
            # as __COL_DEF
            e = VCDStoreElement(r[0], r[1], r[6], r[2], r[3], r[4], r[5])
            ret += [e]

        return tuple(ret)
