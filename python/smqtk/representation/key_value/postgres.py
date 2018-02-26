import logging
import multiprocessing

import six
from six.moves import cPickle as pickle

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value import KeyValueStore, NO_DEFAULT_VALUE
from smqtk.utils.postgres import norm_psql_cmd_string, PsqlConnectionHelper

try:
    import psycopg2
except ImportError as ex:
    logging.getLogger(__name__)\
           .warning("Failed to import psycopg2: %s", str(ex))
    psycopg2 = None


PSQL_TABLE_CREATE_RLOCK = multiprocessing.RLock()


class PostgresKeyValueStore (KeyValueStore):
    """
    PostgreSQL-backed key-value storage.

    This implementation restricts that keys are of a string type. Values are
    serialized (pickle) and stored as bytes in the postgres table.
    """

    class SqlTemplates (object):
        """
        Container for static PostgreSQL queries used by the containing class.
        """

        UPSERT_TABLE_TMPL = norm_psql_cmd_string("""
            CREATE TABLE IF NOT EXISTS {table_name:s} (
              {key_col:s} BYTEA NOT NULL,
              {value_col:s} BYTEA NOT NULL,
              PRIMARY KEY ({key_col:s})
            );
        """)

        SELECT_TMPL = norm_psql_cmd_string("""
            SELECT {query:s} FROM {table_name:s};
        """)

        SELECT_LIKE_TMPL = norm_psql_cmd_string("""
            SELECT {query:s}
              FROM {table_name:s}
             WHERE {key_col:s} like %(key_like)s
        """)

        UPSERT_TMPL = norm_psql_cmd_string("""
            WITH upsert AS (
              UPDATE {table_name:s}
                SET {value_col:s} = %(val)s
                WHERE {key_col:s} = %(key)s
                RETURNING *
              )
            INSERT INTO {table_name:s}
              ({key_col:s}, {value_col:s})
              SELECT %(key)s, %(val)s
                WHERE NOT EXISTS (SELECT * FROM upsert)
        """)

        DELETE_ALL = norm_psql_cmd_string("""
            DELETE FROM {table_name:s}
        """)

    @classmethod
    def is_usable(cls):
        return psycopg2 is not None

    def __init__(self, table_name="data_set",
                 key_col='key', value_col='value', db_name='postgres',
                 db_host=None, db_port=None, db_user=None, db_pass=None,
                 batch_size=1000, pickle_protocol=-1,
                 read_only=False, create_table=True):
        """
        Initialize a PostgreSQL-backed data set instance.

        :param table_name: Name of the table to use.
        :type table_name: str

        :param key_col: Name of the column containing the UUID signatures.
        :type key_col: str

        :param value_col: Name of the table column that will contain
            serialized elements.
        :type value_col: str

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

        :param batch_size: For queries that handle sending or
            receiving many queries at a time, batch queries based on this size.
            If this is None, then no batching occurs.

            The advantage of batching is that it reduces the memory impact for
            queries dealing with a very large number of elements (don't have to
            store the full query for all elements in RAM), but the transaction
            will be some amount slower due to splitting the query into multiple
            transactions.
        :type batch_size: int | None

        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :type pickle_protocol: int

        :param read_only: Only allow read actions against this index.
            Modification actions will throw a ReadOnlyError exceptions.
        :type read_only: bool

        :param create_table: If this instance should try to create the storing
            table before actions are performed against it when not set to be
            read-only. If the configured user does not have sufficient
            permissions to create the table and it does not currently exist, an
            exception will be raised.
        :type create_table: bool

        """
        super(PostgresKeyValueStore, self).__init__()

        self._table_name = table_name
        self._key_col = key_col
        self._value_col = value_col

        self._batch_size = batch_size
        self._pickle_protocol = pickle_protocol
        self._read_only = bool(read_only)
        self._create_table = create_table

        # Checking parameters where necessary
        if self._batch_size is not None:
            self._batch_size = int(self._batch_size)
            assert self._batch_size > 0, \
                "A given batch size must be greater than 0 in size " \
                "(given: %d)." % self._batch_size
        assert -1 <= self._pickle_protocol <= 2, \
            ("Given pickle protocol is not in the known valid range [-1, 2]. "
             "Given: %s." % self._pickle_protocol)

        # helper structure for SQL operations.
        self._psql_helper = PsqlConnectionHelper(
            db_name, db_host, db_port, db_user, db_pass,
            itersize=batch_size,
            table_upsert_lock=PSQL_TABLE_CREATE_RLOCK,
        )

        # Only set table upsert if not read-only.
        if not self._read_only:
            # NOT read-only, so allow table upsert.
            self._psql_helper.set_table_upsert_sql(
                self.SqlTemplates.UPSERT_TABLE_TMPL.format(
                    table_name=self._table_name,
                    key_col=self._key_col,
                    value_col=self._value_col
                )
            )

    @staticmethod
    def _py_to_bin(k):
        """
        Convert a python hashable value into psycopg2.Binary via pickle.

        :param k: Python hashable type.
        :type k: collections.Hashable

        :return: String conversion, aka pickle dump
        :rtype: psycopg2.Binary

        """
        return psycopg2.Binary(pickle.dumps(k))

    @staticmethod
    def _bin_to_py(b):
        """
        Un-"translate" psycopg2.Binary value (buffer) to a python type.

        :param b: buffer from postgres
        :type b: buffer

        :return: Python hashable type.
        :rtype: collections.Hashable

        """
        return pickle.loads(str(b))

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            "table_name": self._table_name,
            "key_col": self._key_col,
            "value_col": self._value_col,

            "db_name": self._psql_helper.db_name,
            "db_host": self._psql_helper.db_host,
            "db_port": self._psql_helper.db_port,
            "db_user": self._psql_helper.db_user,
            "db_pass": self._psql_helper.db_pass,

            "batch_size": self._batch_size,
            "pickle_protocol": self._pickle_protocol,
            "read_only": self._read_only,
            "create_table": self._create_table,
        }

    def __repr__(self):
        """
        Return representative string for this class.

        :return: Representative string for this class.
        :rtype: str

        """
        return super(PostgresKeyValueStore, self).__repr__() \
            % ("table_name: %s, key_col: %s, value_col: %s, "
               "db_name: %s, db_host: %s, db_port: %s, db_user: %s, "
               "db_pass: %s, batch_size: %d, pickle_protocol: %d, "
               "read_only: %s, create_table: %s"
               % (self._table_name, self._key_col, self._value_col,
                  self._psql_helper.db_name, self._psql_helper.db_host,
                  self._psql_helper.db_port, self._psql_helper.db_user,
                  self._psql_helper.db_pass, self._batch_size,
                  self._pickle_protocol, self._read_only, self._create_table))

    def count(self):
        """
        :return: The number of key-value relationships in this store.
        :rtype: int | long
        """
        def cb(cur):
            cur.execute(self.SqlTemplates.SELECT_TMPL.format(
                query='count(%s)' % self._key_col,
                table_name=self._table_name,
            ))
        return list(self._psql_helper.single_execute(cb, True))[0][0]

    def keys(self):
        """
        :return: Iterator over keys in this store.
        :rtype: collections.Iterator[collections.Hashable]
        """
        def cb(cur):
            cur.execute(self.SqlTemplates.SELECT_TMPL.format(
                query=self._key_col,
                table_name=self._table_name,
            ))
        # We can use a named cursor because this is a select statement as well
        # as server table size may be large.
        for r in self._psql_helper.single_execute(cb, True, True):
            # Convert from buffer -> string -> python
            yield self._bin_to_py(r[0])

    def values(self):
        """
        :return: Iterator over values in this store. Values are not guaranteed
            to be in any particular order.
        :rtype: collections.Iterator[object]
        """
        def cb(cur):
            cur.execute(self.SqlTemplates.SELECT_TMPL.format(
                query=self._value_col,
                table_name=self._table_name,
            ))
        for r in self._psql_helper.single_execute(cb, True, True):
            # Convert from buffer -> string -> python
            yield self._bin_to_py(r[0])

    def is_read_only(self):
        """
        :return: True if this instance is read-only and False if it is not.
        :rtype: bool
        """
        return self._read_only

    def has(self, key):
        """
        Check if this store has a value for the given key.

        :param key: Key to check for a value for.
        :type key: collections.Hashable

        :return: If this store has a value for the given key.
        :rtype: bool

        """
        super(PostgresKeyValueStore, self).has(key)

        # Try to select based on given key value. If any rows are returned,
        # there is clearly a key that matches.
        q = self.SqlTemplates.SELECT_LIKE_TMPL.format(
            query='true',
            table_name=self._table_name,
            key_col=self._key_col,
        )

        def cb(cur):
            cur.execute(q, {'key_like': self._py_to_bin(key)})
        return bool(list(self._psql_helper.single_execute(cb, True)))

    def add(self, key, value):
        """
        Add a key-value pair to this store.

        :param key: Key for the value. Must be hashable.
        :type key: collections.Hashable

        :param value: Python object to store.
        :type value: object

        :raises ReadOnlyError: If this instance is marked as read-only.

        :return: Self.
        :rtype: KeyValueStore

        """
        super(PostgresKeyValueStore, self).add(key, value)

        q = self.SqlTemplates.UPSERT_TMPL.format(
            table_name=self._table_name,
            key_col=self._key_col,
            value_col=self._value_col,
        )
        v = {
            'key': self._py_to_bin(key),
            'val': self._py_to_bin(value),
        }

        def cb(cur):
            cur.execute(q, v)

        list(self._psql_helper.single_execute(cb))

    def add_many(self, d):
        """
        Add multiple key-value pairs at a time into this store as represented in
        the provided dictionary `d`.

        :param d: Dictionary of key-value pairs to add to this store.
        :type d: dict[collections.Hashable, object]

        :return: Self.
        :rtype: KeyValueStore

        """
        # Custom override to take advantage of PSQL batching.
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

        q = self.SqlTemplates.UPSERT_TMPL.format(
            table_name=self._table_name,
            key_col=self._key_col,
            value_col=self._value_col,
        )

        # Iterator over transformed inputs into values for statement.
        def val_iter():
            for key, val in six.iteritems(d):
                yield {
                    'key': self._py_to_bin(key),
                    'val': self._py_to_bin(val)
                }

        def cb(cur, v_batch):
            cur.executemany(q, v_batch)

        list(self._psql_helper.batch_execute(val_iter(), cb, self._batch_size))

    def get(self, key, default=NO_DEFAULT_VALUE):
        """
        Get the value for the given key.

        *NOTE:* **Implementing sub-classes are responsible for raising a
        ``KeyError`` where appropriate.**

        :param key: Key to get the value of.
        :type key: collections.Hashable

        :param default: Optional default value if the given key is not present
            in this store. This may be any value except for the
            ``NO_DEFAULT_VALUE`` constant (custom anonymous class instance).
        :type default: object

        :raises KeyError: The given key is not present in this store and no
            default value given.

        :return: Deserialized python object stored for the given key.
        :rtype: object

        """
        q = self.SqlTemplates.SELECT_LIKE_TMPL.format(
            query=self._value_col,
            table_name=self._table_name,
            key_col=self._key_col,
        )
        v = {'key_like': self._py_to_bin(key)}

        def cb(cur):
            cur.execute(q, v)

        rows = list(self._psql_helper.single_execute(cb, True))
        # If no rows and no default, raise KeyError.
        if len(rows) == 0:
            if default is NO_DEFAULT_VALUE:
                raise KeyError(key)
            else:
                return default
        return self._bin_to_py(rows[0][0])

    def clear(self):
        """
        Clear this key-value store.

        *NOTE:* **Implementing sub-classes should call this super-method. This
        super method should not be considered a critical section for thread
        safety.**

        :raises ReadOnlyError: If this instance is marked as read-only.

        """
        q = self.SqlTemplates.DELETE_ALL.format(table_name=self._table_name)

        def cb(cur):
            cur.execute(q)

        list(self._psql_helper.single_execute(cb))
