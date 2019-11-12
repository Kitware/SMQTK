import multiprocessing
from collections import defaultdict

import numpy

from smqtk.representation import DescriptorElement
from smqtk.utils.postgres import norm_psql_cmd_string, PsqlConnectionHelper

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

    SELECT_MANY_TMPL = norm_psql_cmd_string("""
        SELECT {uuid_col:s}, {binary_col:s}
          FROM {table_name:s}
          WHERE {type_col:s} = %(type_val)s
            AND {uuid_col:s} IN %(uuids_tuple)s
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

        self._psql_helper = None

    def __getstate__(self):
        """
        Construct serialization state.

        Due to the psql_helper containing a lock, it cannot be serialized.  This
        is OK due to our creation of the helper on demand.  The cost incurred by
        discarding the instance upon serialization is that once deserialized
        elsewhere the helper instance will have to be created.  Since this
        creation post-deserialization only happens once, this is acceptable.

        """
        state = super(PostgresDescriptorElement, self).__getstate__()
        state.update({
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
        })
        return state

    def __setstate__(self, state):
        # Base DescriptorElement parts
        super(PostgresDescriptorElement, self).__setstate__(state)
        # Our parts
        self.table_name = state['table_name']
        self.uuid_col = state['uuid_col']
        self.type_col = state['type_col']
        self.binary_col = state['binary_col']
        self.create_table = state['create_table']
        self.db_name = state['db_name']
        self.db_host = state['db_host']
        self.db_port = state['db_port']
        self.db_user = state['db_user']
        self.db_pass = state['db_pass']
        self._psql_helper = None

    @classmethod
    def _create_psql_helper(
            cls, db_name, db_host, db_port, db_user, db_pass, table_name,
            uuid_col, type_col, binary_col, itersize=1000, create_table=True):
        """
        Internal helper function for creating PSQL connection helpers for class
        instances.

        :param db_name: The name of the database to connect to.
        :type db_name: str

        :param db_host: Host address of the Postgres server. If None, we
            assume the server is on the local machine and use the UNIX socket.
            This might be a required field on Windows machines (not tested yet).
        :type db_host: str | None

        :param db_port: Port the Postgres server is exposed on. If None, we
            assume the default port (5423).
        :type db_port: int | None

        :param db_user: Postgres user to connect as. If None, postgres
            defaults to using the current accessing user account name on the
            operating system.
        :type db_user: str | None

        :param db_pass: Password for the user we're connecting as. This may be
            None if no password is to be used.
        :type db_pass: str | None

        :param table_name: String label of the database table to use.
        :type table_name: str

        :param uuid_col: The column label for descriptor UUID storage
        :type uuid_col: str

        :param type_col: The column label for descriptor type string storage.
        :type type_col: str

        :param binary_col: The column label for descriptor vector binary
            storage.
        :type binary_col: str

        :param itersize: Number of records fetched per network round trip when
            iterating over a named cursor. This parameter only does anything if
            a named cursor is used.
        :type itersize: int

        :param create_table: Whether to try to create the storing table before
            returning the connection helper. If the configured user does not
            have sufficient permissions to create the table and it does not
            currently exist, an exception will be raised.
        :type create_table: bool

        :return: PsqlConnectionHelper utility.
        :rtype: PsqlConnectionHelper
        """
        helper = PsqlConnectionHelper(
            db_name, db_host, db_port, db_user, db_pass,
            itersize=itersize, table_upsert_lock=PSQL_TABLE_CREATE_RLOCK
        )

        if create_table:
            helper.set_table_upsert_sql(
                cls.UPSERT_TABLE_TMPL.format(
                    table_name=table_name,
                    type_col=type_col,
                    uuid_col=uuid_col,
                    binary_col=binary_col
                )
            )

        return helper

    def _get_psql_helper(self):
        """
        Internal method to create on demand the PSQL connection helper class.
        :return: PsqlConnectionHelper utility.
        :rtype: PsqlConnectionHelper
        """
        if self._psql_helper is None:
            # Only using a transport iteration size of 1 since this element is
            # only meant to refer to a single entry in the associated table.
            self._psql_helper = self._create_psql_helper(
                self.db_name, self.db_host, self.db_port, self.db_user,
                self.db_pass, self.table_name, self.type_col, self.uuid_col,
                self.binary_col, itersize=1, create_table=self.create_table
            )
        return self._psql_helper

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

        def cb(cursor):
            cursor.execute(q_select, q_select_values)

        # Should either yield one or zero rows.
        psql_helper = self._get_psql_helper()
        return bool(list(psql_helper.single_execute(
            cb, yield_result_rows=True
        )))

    def vector(self):
        """
        Return this element's vector, or None if we don't have one.

        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.ndarray or None

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

        # query execution callback
        # noinspection PyProtectedMember
        def cb(cursor):
            # type: (psycopg2._psycopg.cursor) -> None
            cursor.execute(q_select, q_select_values)

        # This should only fetch a single row.  Cannot yield more than one due
        # use of primary keys.
        psql_helper = self._get_psql_helper()
        r = list(psql_helper.single_execute(cb, yield_result_rows=True))
        if not r:
            return None
        else:
            b = r[0][0]
            v = numpy.frombuffer(b, self.ARRAY_DTYPE)
            return v

    @classmethod
    def _sql_vector_query_options(cls, descriptor):
        """
        Internal helper method to construct tuple of options used to construct
        sql query for given descriptor's vector.

        :return: Tuple of elements used to construct a SQL query
        :rtype: tuple
        """
        return (
            descriptor.db_name,
            descriptor.db_host,
            descriptor.db_port,
            descriptor.db_user,
            descriptor.db_pass,
            descriptor.table_name,
            descriptor.type_col,
            descriptor.uuid_col,
            descriptor.binary_col,
            descriptor.type()
        )

    @classmethod
    def _get_many_vectors(cls, descriptors):
        """
        Internal method to be overridden by subclasses to return many vectors
        associated with given descriptors.

        :note: Returned vectors are *not* guaranteed to be returned in the
            order they are requested. Missing vectors may be returned as None
            or omitted entirely from results. The wrapper function
            `get_many_vectors` handles re-ordering as necessary and insertion
            of None for missing values.

        :param descriptors: Iterable of descriptors to query for.
        :type descriptors: collections.Iterable[
            smqtk.representation.descriptor_element.DescriptorElement]

        :return: Iterator of tuples containing the descriptor uuid and the
            vector associated with the given descriptors or None if the
            descriptor has no associated vector
        :rtype: collections.Iterable[
            tuple[collections.Hashable, Union[numpy.ndarray, None]]]
        """
        batch_dictionary = defaultdict(list)
        # For each given descriptor...
        for descriptor_ in descriptors:
            # Extract options for constructing SQL query used to
            # retrieve descriptor vectors
            batch_dictionary[
                cls._sql_vector_query_options(descriptor_)
            ].append(descriptor_.uuid())

        # For each unique set of SQL query options...
        for query_options, uuids in batch_dictionary.items():
            psql_helper = cls._create_psql_helper(
                *query_options[:-1], create_table=False)

            sql_query = cls.SELECT_MANY_TMPL.format(
                table_name=query_options[5],
                type_col=query_options[6],
                uuid_col=query_options[7],
                binary_col=query_options[8],
            )

            sql_values = {
                "type_val": query_options[9],
                "uuids_tuple": tuple(uuids)
            }

            def query_callback(cursor):
                cursor.execute(sql_query, sql_values)

            # Perform a SQL query to retrieve all vectors in this batch
            sql_return = psql_helper.single_execute(
                query_callback, yield_result_rows=True
            )

            # Construct numpy array from buffer and return uuid, vector pairs
            for uuid, vector_buffer in sql_return:
                yield (uuid, numpy.frombuffer(vector_buffer, cls.ARRAY_DTYPE))

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
        :type new_vec: numpy.ndarray

        :returns: Self.
        :rtype: PostgresDescriptorElement

        """
        if not isinstance(new_vec, numpy.ndarray):
            new_vec = numpy.copy(new_vec)

        if new_vec.dtype != self.ARRAY_DTYPE:
            try:
                new_vec = new_vec.astype(self.ARRAY_DTYPE)
            except TypeError:
                raise ValueError("Could not convert input to a vector of type "
                                 "%s." % self.ARRAY_DTYPE)

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

        # query execution callback
        # noinspection PyProtectedMember
        def cb(cursor):
            # type: (psycopg2._psycopg.cursor) -> None
            cursor.execute(q_upsert, q_upsert_values)

        # No return but need to force iteration.
        psql_helper = self._get_psql_helper()
        list(psql_helper.single_execute(cb, yield_result_rows=False))
        return self
