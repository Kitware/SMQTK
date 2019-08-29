"""
Data element for storing bytes in PostgreSQL.

Note about PSQL and clusters
----------------------------
To create a cluster::

    $ sudo pg_createcluster -u $(whoami) 9.5 squadx

This does not start the cluster, which should be started with the following::

    $ pg_ctlcluster 9.5 squadx start

Note that we do not provide the ``--start`` flag to the pg_createcluster
command.  This because the postgresql server fails to start or stay active when
using the recommended systemctl-based commands.  Instead, pg_ctlcluster should
be used to start/stop the database.

"""
import hashlib
from threading import RLock

import six

from smqtk.representation import DataElement
from smqtk.utils.postgres import norm_psql_cmd_string, PsqlConnectionHelper

# Try to import required modules
try:
    import psycopg2
except ImportError:
    psycopg2 = None


# Lock for data element create-table functionality
GLOBAL_PSQL_TABLE_CREATE_RLOCK = RLock()


class PostgresDataElement (DataElement):
    """
    Data element bytes stored in PostgreSQL database.

    Storage table should have three columns for the following components:
    - data SHA1 (effective UID)
    - data content-type / MIMETYPE
    - data bytes

    Efficient connection pooling may be achieved via external utilities like
    PGBounder.

    Due to the use of the "ON CONFLICT" clause in upserting data, this
    implementation requires at least PostgreSQL version 9.5 or greater.

    """

    # SHA1 checksum of 0-length data (empty bytes)
    EMPTY_SHA = hashlib.sha1(b'').hexdigest()

    class CommandTemplates (object):
        """ Encapsulation of command templates. """

        # Upsert table for storage if desired
        #
        # Format params:
        # - table_name
        # - id_col
        # - sha1_col
        # - mime_col
        # - byte_col
        UPSERT_TABLE = norm_psql_cmd_string("""
            CREATE TABLE IF NOT EXISTS {table_name:s} (
              {id_col:s}   TEXT NOT NULL,
              {sha1_col:s} TEXT NOT NULL,
              {mime_col:s} TEXT NOT NULL,
              {byte_col:s} BYTEA NOT NULL,
              PRIMARY KEY ({id_col:s})
            );
        """)

        # Select ``col`` for a given entry ID.
        #
        # Query Format params:
        # - col
        # - table_name
        # - id_col
        #
        # Value params:
        # - id_val
        SELECT = norm_psql_cmd_string("""
            SELECT {col:s}
              FROM {table_name:s}
              WHERE {id_col:s} = %(id_val)s
            ;
        """)

        # Upsert content-type/data for a uid
        #
        # Query Format params:
        # - table_name
        # - id_col
        # - sha1_col
        # - mime_col
        # - byte_col
        #
        # Value params:
        # - id_val
        # - sha1_val
        # - mime_val
        # - byte_val
        #
        # SQL format from:
        #   https://hashrocket.com/blog/posts/upsert-records-with-postgresql-9-5
        #
        UPSERT_DATA = norm_psql_cmd_string("""
            INSERT INTO {table_name:s} ({id_col:s}, {sha1_col:s}, {mime_col:s}, {byte_col:s})
                VALUES ( %(id_val)s, %(sha1_val)s, %(mime_val)s, %(byte_val)s )
                ON CONFLICT ({id_col:s})
                    DO UPDATE
                        SET ({sha1_col:s}, {mime_col:s}, {byte_col:s})
                          = (EXCLUDED.{sha1_col:s}, EXCLUDED.{mime_col:s}, EXCLUDED.{byte_col:s})
            ;
        """)

        # Same as ``UPSERT_DATA`` but does not set the mimetype on an update.
        # This is meant to atomically update the byte data without changing the
        # existing mimetype.
        UPSERT_DATA_NO_MIME = norm_psql_cmd_string("""
            INSERT INTO {table_name:s} ({id_col:s}, {sha1_col:s}, {mime_col:s}, {byte_col:s})
                VALUES ( %(id_val)s, %(sha1_val)s, %(mime_val)s, %(byte_val)s )
                ON CONFLICT ({id_col:s})
                    DO UPDATE
                        SET ({sha1_col:s}, {byte_col:s})
                          = (EXCLUDED.{sha1_col:s}, EXCLUDED.{byte_col:s})
            ;
        """)

    @classmethod
    def is_usable(cls):
        if psycopg2 is None:
            cls.get_logger().warning("Not usable. "
                                     "Requires the psycopg2 module.")
            return False
        return True

    def __init__(self, element_id, content_type=None,
                 table_name="psql_data_elements", id_col="id",
                 sha1_col="sha1", mime_col="mime", byte_col="bytes",
                 db_name="postgres", db_host="/tmp", db_port=5433, db_user=None,
                 db_pass=None, read_only=False,
                 create_table=True):
        """
        Create a new PostgreSQL-based data element.

        If the tabled mapped to the provided ``table_name`` already exists, we
        expect the provided columns to match the following types:
        - ``id_col`` is expected to be TEXT
        - ``sha1_col`` is expected to be TEXT
        - ``type_col`` is expected to be TEXT
        - ``byte_col`` is expected to be BYTEA

        Default database connection parameters are assuming the use of a
        non-default, non-postgres-user cluster where the current user's name is
        equivalent to a valid role in the database.

        :param element_id: ID to reference a specific data element row in the
            table.  This is required in the same way that a path is required to
            point to a file on a filesystem.
        :type element_id: str

        :param content_type: Expected mime-type of byte data set to this
            element.  This only affects setting the mime-type field when setting
            new bytes.  ``content_type()`` will always reflect what is stored in
            the backend, or lack there-of.

            If this mime-type differs from an existing stored value,
            this mime-type will overwrite the stored value on the next call to
            ``set_bytes``.  If this is None and there is no mime-type already
            set in the database, no mime-type will be set on the next
            ``set_bytes`` call.
        :type content_type: str | None

        :param table_name: String label of the table in the database to interact
            with.
        :type table_name: str

        :param id_col: Name of the element ID column in ``table_name``.
        :type id_col: str

        :param sha1_col: Name of the SHA1 column in ``table_name``.
        :type sha1_col: str

        :param mime_col: Name of the MIMETYPE column in ``table_name``.
        :type mime_col: str

        :param byte_col: Name of the column storing byte data in ``table_name``.
        :type byte_col: str

        :param db_host: Host address of the PostgreSQL server. If None, we
            assume the server is on the local machine and use the UNIX socket.
            This might be a required field on Windows machines (not tested yet).
        :type db_host: str | None

        :param db_port: Port the Postgres server is exposed on. If None, we
            assume a default port (5433).
        :type db_port: int | None

        :param db_name: The name of the database to connect to.
        :type db_name: str

        :param db_user: Postgres user to connect as. If None, postgres
            defaults to using the current accessing user account name on the
            operating system.
        :type db_user: str | None

        :param db_pass: Password for the user we're connecting as. This may be
            None if no password is to be used.
        :type db_pass: str | None

        :param read_only: Only allow reading of this data.  Modification actions
            will throw a ReadOnlyError exceptions.
        :type read_only: bool

        :param create_table: If this instance should try to create the storing
            table before actions are performed against it. If the configured
            user does not have sufficient permissions to create the table and it
            does not currently exist, an exception will be raised.
        :type create_table: bool

        """
        super(PostgresDataElement, self).__init__()

        if not isinstance(element_id, six.string_types):
            raise ValueError("Element ID should be a string type.")

        self._element_id = element_id
        self._content_type = content_type
        self._table_name = table_name

        self._id_col = id_col
        self._sha1_col = sha1_col
        self._mime_col = mime_col
        self._byte_col = byte_col

        self._read_only = read_only
        self._create_table = create_table

        # itersize is hard-coded because a single-element perspective should
        # only be retrieving one row at a time.
        self._psql_helper = PsqlConnectionHelper(db_name, db_host, db_port,
                                                 db_user, db_pass, 10,
                                                 GLOBAL_PSQL_TABLE_CREATE_RLOCK)

        # Set table creation SQL in helper
        if not self._read_only:
            self._psql_helper.set_table_upsert_sql(
                self.CommandTemplates.UPSERT_TABLE.format(
                    table_name=self._table_name,
                    id_col=self._id_col,
                    sha1_col=self._sha1_col,
                    mime_col=self._mime_col,
                    byte_col=byte_col,
                )
            )

    def __repr__(self):
        return "{:s}[id=\"{:s}\"]" \
            .format(self.__class__.__name__, self._element_id)

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            "element_id": self._element_id,
            "table_name": self._table_name,
            "id_col": self._id_col,
            "sha1_col": self._sha1_col,
            "mime_col": self._mime_col,
            "byte_col": self._byte_col,

            "db_name": self._psql_helper.db_name,
            "db_host": self._psql_helper.db_host,
            "db_port": self._psql_helper.db_port,
            "db_user": self._psql_helper.db_user,
            "db_pass": self._psql_helper.db_pass,

            "read_only": self._read_only,
            "create_table": self._create_table,
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        q = self.CommandTemplates.SELECT.format(
            col=self._mime_col,
            table_name=self._table_name,
            id_col=self._id_col,
        )
        v = dict(
            id_val=self._element_id
        )

        def cb(cursor):
            """
            :type cursor: psycopg2._psycopg.cursor
            """
            cursor.execute(q, v)

        r = list(self._psql_helper.single_execute(cb, yield_result_rows=True))
        if not r:
            return None
        elif len(r) > 1:
            raise RuntimeError("Somehow found multiple entries for the same"
                               "element ID (there should only be one).")
        return r[0][0]

    def is_empty(self):
        """
        Check if this element contains no bytes.

        The intent of this method is to quickly check if there is any data
        behind this element, ideally without having to read all/any of the
        underlying data.

        :return: If this element contains 0 bytes.
        :rtype: bool

        """
        q = self.CommandTemplates.SELECT.format(
            col="octet_length(%s)" % self._byte_col,
            table_name=self._table_name,
            id_col=self._id_col,
        )
        v = dict(
            id_val=self._element_id
        )

        def cb(cursor):
            """
            :type cursor: psycopg2._psycopg.cursor
            """
            cursor.execute(q, v)

        r = list(self._psql_helper.single_execute(cb, yield_result_rows=True))
        if not r:
            # No rows returned, meaning not entry for our element ID and no
            # bytes stored.
            return True
        elif len(r) > 1:
            raise RuntimeError("Somehow found multiple entries for the same"
                               "element ID (there should only be one).")

        num_bytes = int(r[0][0])
        if num_bytes == 0:
            # There was an entry, but the number of bytes stored was zero.
            return True
        else:
            # Non-zero number of bytes stored.
            return False

    def sha1(self):
        """
        Get the SHA1 checksum of this element's binary content.

        :return: SHA1 hex checksum of the data content.
        :rtype: str
        """
        q = self.CommandTemplates.SELECT.format(
            col=self._sha1_col,
            table_name=self._table_name,
            id_col=self._id_col,
        )
        v = dict(
            id_val=self._element_id,
        )

        def cb(cursor):
            """
            :type cursor: psycopg2._psycopg.cursor
            """
            cursor.execute(q, v)

        r = list(self._psql_helper.single_execute(cb, yield_result_rows=True))
        if not r:
            # no rows for element ID, so no bytes. Return SHA1 of empty string
            return self.EMPTY_SHA
        return r[0][0]

    def get_bytes(self):
        """
        :return: Get the bytes for this data element.
        :rtype: bytes
        """
        q = self.CommandTemplates.SELECT.format(
            col=self._byte_col,
            table_name=self._table_name,
            id_col=self._id_col,
        )
        v = dict(
            id_val=self._element_id
        )

        def cb(cursor):
            """
            :type cursor: psycopg2._psycopg.cursor
            """
            cursor.execute(q, v)

        r = list(self._psql_helper.single_execute(cb, yield_result_rows=True))
        if not r or len(r[0][0]) == 0:
            # No returned rows for element ID or if no bytes are stored.
            return bytes()
        else:
            return bytes(r[0][0])

    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        return not self._read_only

    def set_bytes(self, b):
        """
        Set bytes to this data element.

        Not all implementations may support setting bytes (check ``writable``
        method return).

        This base abstract method should be called by sub-class implementations
        first. We check for mutability based on ``writable()`` method return.

        :param b: bytes to set.
        :type b: byte

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.

        """
        super(PostgresDataElement, self).set_bytes(b)

        b_sha1 = hashlib.sha1(b).hexdigest()

        # TODO: Fallback to ``content_type()`` return if none provided in self.
        if self._content_type:
            # We have a content/mime type override as specified at element
            # construction.
            b_mimetype = self._content_type
            q_tmpl = self.CommandTemplates.UPSERT_DATA
        else:
            # Leave the mimetype alone or set an empty mimetype (none specified
            # at construction).
            b_mimetype = ""
            q_tmpl = self.CommandTemplates.UPSERT_DATA_NO_MIME

        q = q_tmpl.format(
            table_name=self._table_name,
            id_col=self._id_col,
            sha1_col=self._sha1_col,
            mime_col=self._mime_col,
            byte_col=self._byte_col,
        )
        v = dict(
            id_val=self._element_id,
            sha1_val=b_sha1,
            mime_val=b_mimetype,
            byte_val=psycopg2.Binary(b)
        )

        def cb(cursor):
            """
            :type cursor: psycopg2._psycopg.cursor
            """
            # TODO: Could be smart here and only update if content-type/byte
            #       data differs while keeping a row-lock between queries.
            cursor.execute(q, v)

        list(self._psql_helper.single_execute(cb))
