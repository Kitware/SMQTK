import multiprocessing

import numpy

from smqtk.representation import DescriptorElement
from smqtk.utils.postgres import norm_psql_cmd_string

# Try to import required modules
try:
    import psycopg2
except ImportError:
    psycopg2 = None


PSQL_TABLE_CREATE_RLOCK = multiprocessing.RLock()


# noinspection SqlNoDataSourceInspection
class PostgresDescriptorElement (DescriptorElement):
    """
    Descriptor element whose vector is stored in a Postgres database.

    We assume we will work with a Postgres version of at least 9.4 (due to
    versions tested).

    Efficient connection pooling may be achieved via external utilities like
    PGBounder.

    """

    ARRAY_DTYPE = numpy.float64

    UPSERT_TABLE_TMPL = norm_psql_cmd_string("""
        CREATE TABLE IF NOT EXISTS {table_name:s} (
          {type_col:s} TEXT NOT NULL,
          {uuid_col:s} TEXT NOT NULL,
          {binary_col:s} BYTEA NOT NULL,
          PRIMARY KEY ({type_col:s}, {uuid_col:s})
        );
    """)

    SELECT_TMPL = norm_psql_cmd_string("""
        SELECT {binary_col:s}
          FROM {table_name:s}
          WHERE {type_col:s} = %(type_val)s
            AND {uuid_col:s} = %(uuid_val)s
        ;
    """)

    UPSERT_TMPL = norm_psql_cmd_string("""
        WITH upsert AS (
          UPDATE {table_name:s}
            SET {binary_col:s} = %(binary_val)s
            WHERE {type_col:s} = %(type_val)s
              AND {uuid_col:s} = %(uuid_val)s
            RETURNING *
          )
        INSERT INTO {table_name:s} ({type_col:s}, {uuid_col:s}, {binary_col:s})
          SELECT %(type_val)s, %(uuid_val)s, %(binary_val)s
            WHERE NOT EXISTS (SELECT * FROM upsert);
    """)

    @classmethod
    def is_usable(cls):
        if psycopg2 is None:
            cls.get_logger().warning("Not usable. Requires psycopg2 module")
            return False
        return True

    def __init__(self, type_str, uuid,
                 table_name='descriptors',
                 uuid_col='uid', type_col='type_str', binary_col='vector',
                 db_name='postgres', db_host=None, db_port=None, db_user=None,
                 db_pass=None, create_table=True):
        """
        Initialize new PostgresDescriptorElement attached to some database
        credentials.

        We require that storage tables treat uuid AND type string columns as
        primary keys. The type and uuid columns should be of the 'text' type.
        The binary column should be of the 'bytea' type.

        Default argument values assume a local PostgreSQL database with a table
        created via the
        ``etc/smqtk/postgres/descriptor_element/example_table_init.sql``
        file (relative to the SMQTK source tree or install root).

        NOTES:
            - Not all uuid types used here are necessarily of the ``uuid.UUID``
              type, thus the recommendation to use a ``text`` type for the
              column. For certain specific use cases they may be proper
              ``uuid.UUID`` instances or strings, but this cannot be generally
              assumed.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: Unique ID reference of the descriptor.
        :type uuid: collections.Hashable

        :param table_name: String label of the database table to use.
        :type table_name: str

        :param uuid_col: The column label for descriptor UUID storage
        :type uuid_col: str

        :param type_col: The column label for descriptor type string storage.
        :type type_col: str

        :param binary_col: The column label for descriptor vector binary
            storage.
        :type binary_col: str

        :param db_host: Host address of the Postgres server. If None, we
            assume the server is on the local machine and use the UNIX socket.
            This might be a required field on Windows machines (not tested yet).
        :type db_host: str | None

        :param db_port: Port the Postgres server is exposed on. If None, we
            assume the default port (5423).
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

        :param create_table: If this instance should try to create the storing
            table before actions are performed against it. If the configured
            user does not have sufficient permissions to create the table and it
            does not currently exist, an exception will be raised.
        :type create_table: bool

        """
        super(PostgresDescriptorElement, self).__init__(type_str, uuid)

        self.table_name = table_name
        self.uuid_col = uuid_col
        self.type_col = type_col
        self.binary_col = binary_col
        self.create_table = create_table

        self.db_name = db_name
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_pass = db_pass

    def get_config(self):
        return {
            "table_name": self.table_name,
            "uuid_col": self.uuid_col,
            "type_col": self.type_col,
            "binary_col": self.binary_col,
            "create_table": self.create_table,

            "db_name": self.db_name,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_user": self.db_user,
            "db_pass": self.db_pass,
        }

    def _get_psql_connection(self):
        """
        :return: A new connection to the configured database
        :rtype: psycopg2._psycopg.connection
        """
        return psycopg2.connect(
            database=self.db_name,
            user=self.db_user,
            password=self.db_pass,
            host=self.db_host,
            port=self.db_port,
        )

    def _ensure_table(self, cursor):
        """
        Execute on psql connector cursor the table create-of-not-exists query.

        :param cursor: Connection active cursor.

        """
        if self.create_table:
            q_table_upsert = self.UPSERT_TABLE_TMPL.format(**dict(
                table_name=self.table_name,
                type_col=self.type_col,
                uuid_col=self.uuid_col,
                binary_col=self.binary_col,
            ))
            with PSQL_TABLE_CREATE_RLOCK:
                cursor.execute(q_table_upsert)
                cursor.connection.commit()

    def has_vector(self):
        """
        Check if the target database has a vector for our keys.

        This also returns True if we have a cached vector since there must have
        been a source vector to draw from if there is a cache of it.

        If a vector is cached, this resets the cache expiry timeout.

        :return: Whether or not this container current has a descriptor vector
            stored.
        :rtype: bool

        """
        # Very similar to vector query, but replacing vector binary return with
        # a true/null return. Save a little bit of time compared to testing
        # vector return.
        # OLD: return self.vector() is not None

        # Using static value 'true' for binary "column" to reduce data return
        # volume.
        q_select = self.SELECT_TMPL.format(**{
            'binary_col': 'true',
            'table_name': self.table_name,
            'type_col': self.type_col,
            'uuid_col': self.uuid_col,
        })
        q_select_values = {
            "type_val": self.type(),
            "uuid_val": str(self.uuid())
        }

        conn = self._get_psql_connection()
        cur = conn.cursor()

        try:
            self._ensure_table(cur)
            cur.execute(q_select, q_select_values)
            r = cur.fetchone()
            # For server cleaning (e.g. pgbouncer)
            conn.commit()
            return bool(r)
        except:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def vector(self):
        """
        Return this element's vector, or None if we don't have one.

        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None

        """
        q_select = self.SELECT_TMPL.format(**{
            "binary_col": self.binary_col,
            "table_name": self.table_name,
            "type_col": self.type_col,
            "uuid_col": self.uuid_col,
        })
        q_select_values = {
            "type_val": self.type(),
            "uuid_val": str(self.uuid())
        }

        conn = self._get_psql_connection()
        cur = conn.cursor()
        try:
            self._ensure_table(cur)
            cur.execute(q_select, q_select_values)

            r = cur.fetchone()
            conn.commit()

            if not r:
                return None
            else:
                b = r[0]
                v = numpy.frombuffer(b, self.ARRAY_DTYPE)
                return v
        except:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        If we are configured to use caching, and one has not been cached yet,
        then we cache the vector and start a thread to monitor access times and
        to remove the cache if the access timeout has expired.

        If a vector was already cached, this new vector replaces the old one,
        the vector database-side is replaced, and the cache expiry timeout is
        reset.

        :raises ValueError: ``new_vec`` was not a numpy ndarray.

        :param new_vec: New vector to contain. This must be a numpy array.
        :type new_vec: numpy.core.multiarray.ndarray

        """
        if not isinstance(new_vec, numpy.core.multiarray.ndarray):
            raise ValueError("Input array for setting was not a numpy.ndarray! "
                             "(given: %s)" % type(new_vec))

        if new_vec.dtype != self.ARRAY_DTYPE:
            new_vec = new_vec.astype(self.ARRAY_DTYPE)

        q_upsert = self.UPSERT_TMPL.strip().format(**{
            "table_name": self.table_name,
            "binary_col": self.binary_col,
            "type_col": self.type_col,
            "uuid_col": self.uuid_col,
        })
        q_upsert_values = {
            "binary_val": psycopg2.Binary(new_vec),
            "type_val": self.type(),
            "uuid_val": str(self.uuid()),
        }

        conn = self._get_psql_connection()
        cur = conn.cursor()
        try:
            self._ensure_table(cur)
            cur.execute(q_upsert, q_upsert_values)
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
