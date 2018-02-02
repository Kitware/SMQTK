import logging
from multiprocessing import RLock
import uuid

from smqtk.utils import SmqtkObject

try:
    import psycopg2
except ImportError as ex:
    logging.getLogger(__name__)\
           .warning("Failed to import psycopg2: %s", str(ex))
    psycopg2 = None


GLOBAL_PSQL_TABLE_CREATE_RLOCK = RLock()


def norm_psql_cmd_string(s):
    """
    Simple function to reduce down a multi-line string written for readability
    to a single line.

    :param s: Single or multi-line string with inconsistant white-space.
    :type s: str

    :return: Single-line string.
    :rtype: str

    """
    return ' '.join(s.split())


class PsqlConnectionHelper (SmqtkObject):
    """
    Helper class for things that interact with a PostgreSQL database.
    """

    def __init__(self, db_name='postgres', db_host=None, db_port=None,
                 db_user=None, db_pass=None,
                 itersize=1000,
                 table_upsert_lock=GLOBAL_PSQL_TABLE_CREATE_RLOCK):
        """
        Create a new helper instance for a set of database connection
        parameters.

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

        :param itersize: Number of records fetched per network round trip when
            iterating over a named cursor. This parameter only does anything if
            a named cursor is used.
        :type itersize: int

        :param table_upsert_lock: Lock to guard table upsertion.
        :type table_upsert_lock: multiprocessing.synchronize.RLock
        """
        # Stop construction if psycopg2 is not defined.
        if psycopg2 is None:
            raise RuntimeError("Cannot construct a %s if the psycopg2 module "
                               "is not importable" % self.__class__.__name__)
        self.db_name = db_name
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_pass = db_pass

        self.itersize = itersize

        self.table_upsert_lock = table_upsert_lock
        self.table_upsert_sql = None

    def get_psql_connection(self):
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

    @staticmethod
    def get_unique_cursor_name():
        """
        Return a string to use as a cursor name.

        Named cursors are used to allow for server-side iteration for SELECT
        and VALUES commands. This lessens the load on the machine making the
        query.

        :return: New cursor name string with a unique, random UUID embedded.
        :rtype: str
        """
        ruuid = str(uuid.uuid4()).replace('-', '')
        return "smqtk_postgres_cursor_%s" % ruuid

    def set_table_upsert_sql(self, s):
        """
        SQL optional statement to upsert a table in the database before
        executing statements. If this is not set, table upsertion will not
        occur.

        ``None`` may be to disable upsertion.

        :param s: String SQL statement or ``None``.
        :type s: str | None

        """
        self.table_upsert_sql = s

    def ensure_table(self, cursor):
        """
        Execute on a PSQL connection's cursor the set table upsert command if
        set.

        This method does nothing if no upsert command is set.

        :param cursor: Connection active cursor
        """
        if self.table_upsert_sql is not None:
            with self.table_upsert_lock:
                cursor.execute(self.table_upsert_sql)
                cursor.connection.commit()

    def single_execute(self, cursor_callback, yield_result_rows=False,
                       named=False):
        """
        Perform a single SQL execution in  a new database connection. Handles
        connection/cursor acquisition and release.

        After a successful call to the given callback all result rows are
        yielded, a connection commit is performed in order to finalize any
        changes and to flush the connection (e.g. for pgbouncer).

        Due to this method optionally yielding values, calling this returns a
        generator. This must be iterated over for anything to occur even if
        nothing is to be actively yielded (e.g. execute callback only performs
        writes).

        :param cursor_callback: Function that takes a single positional argument
            that is the active cursor on a new database connection. This
            function should perform any ``execute`` actions on the provided
            cursor required.

            If this callback raises an exception, a rollback is performed on the
            connection and the connection is closed, passing the exception
            forward.
        :type cursor_callback: (psycopg2._psycopg.cursor) -> None

        :param yield_result_rows: Optionally yield rows from each batch
            execution. False by default. If False, nothing is returned.
        :type yield_result_rows: bool

        :param named: If a named cursor should be created, creating a
            server-side cursor. This is only compatible with executions of
            SELECT or VALUES commands.
        :type named: bool

        :return: Iterator over result rows if ``yield_result_rows`` is True.
        :rtype: __generator

        """
        conn = self.get_psql_connection()

        # Optionally create a named cursor to allow server-side iteration. This
        # is required in order to not pull the whole table into memory.
        cursor_name = None
        if named:
            cursor_name = self.get_unique_cursor_name()

        try:
            with conn:
                with conn.cursor() as cur:
                    self.ensure_table(cur)
                with conn.cursor(cursor_name) as cur:
                    # Number of records ``iter(cur)`` must fetch per network
                    # round trip.
                    # - This only maters if the cursor is a named cursor
                    #   (server-side)
                    cur.itersize = self.itersize
                    cursor_callback(cur)

                    # Iterate over returned cursor rows.
                    if yield_result_rows:
                        for r in cur:
                            yield r

                # For server cleaning (e.g. pgbouncer)
                conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            # conn.__exit__ doesn't close connection, just the transaction
            conn.close()

    def batch_execute(self, iterable, cursor_callback, batch_size,
                      yield_result_rows=False, named=False):
        """
        Batch the given iterable into ``batch_size`` chunks, calling the given
        ``cursor_callback`` with each batch.

        This is generally intended for when the use of the
        ``cursor.executemany`` function is appropriate for statements other than
        SELECT or VALUES.

        When performing a query with a large number of component elements, the
        time between performing the query and getting the first response can be
        long. Additionally, the whole query would need to be constructed first
        which may take a long time or consume a large amount of memory for large
        iterables.

        Splitting an iterable of elements up into batches lessens the
        construction time and memory foot-print of each individual query. This
        in turn improves the response time for individual queries to the server
        as well as keeping the local memory requirements static. A fast response
        time means that results can be iterated out and errors can be revealed
        more quickly.

        This method handles creating a new connection and cursor, ensuring the
        a table's existence (if that query was set) and pulling together batches
        of the given size from the provided opaque iterable.

        After a successful call to the given callback all result rows are
        yielded. After all calls to the given callback are performed, a
        connection commit is performed in order to finalize any changes and to
        flush the connection (e.g. for pgbouncer).

        Due to this method optionally yielding values, calling this returns a
        generator. This must be iterated over for anything to occur even if
        nothing is to be actively yielded.

        :param iterable: Iterable of elements to batch.
        :type iterable: collections.Iterable

        :param cursor_callback: Function that takes two positional arguments:
            the cursor object for the database connection and the current batch
            elements (list-type). The given batch of elements may be of up to
            size ``batch_size`` or less (the "tail" batch). This function should
            perform ``executemany`` actions on the provided cursor for the given
            batch. Results are optionally yielded af this call successfully
            returns.

            If this callback raises an exception, a rollback is performed on the
            connection and the connection is closed, passing the exception
            forward.
        :type cursor_callback: (psycopg2._psycopg.cursor, list) -> None

        :param batch_size: Batch size limit when pulling elements from the input
            iterable. May be a positive integer, 0 or None. A value of 0 or None
            means that no batching occurs and all elements from the iterable are
            collected.

            Once this number of elements are collected from the iterable, the
            callback is called with the collected batch of elements.
        :type batch_size: int | None

        :param yield_result_rows: Optionally yield rows from each batch
            execution. False by default.
        :type yield_result_rows: bool

        :param named: If a named cursor should be created, creating a
            server-side cursor. This is only compatible with executions of
            SELECT or VALUES commands.
        :type named: bool

        :return: Iterator over result rows if ``yield_result_rows`` is True,
            otherwise None.
        :rtype: __generator | None

        """
        if batch_size is None:
            batch_size = 0
        if batch_size < 0:
            raise ValueError("Cannot have a batch size less than 0 "
                             "(given: %s)." % batch_size)

        self._log.debug("starting multi operation (batch_size: %s)", batch_size)

        # Lazy initialize -- only if there are elements to iterate over
        #: :type: None | psycopg2._psycopg.connection
        conn = None

        # Create a named cursor to allow server-side iteration. This is
        # required in order to not pull the whole table into memory.
        cursor_name = None
        if named:
            cursor_name = self.get_unique_cursor_name()

        try:
            batch = []
            i = 0
            for e in iterable:
                if conn is None:
                    conn = self.get_psql_connection()

                batch.append(e)

                if batch_size and len(batch) >= batch_size:
                    i += 1
                    self._log.debug('-- batch %d (size: %d)', i, len(batch))

                    with conn:
                        with conn.cursor() as cur:
                            self.ensure_table(cur)
                        with conn.cursor(cursor_name) as cur:
                            cur.itersize = self.itersize
                            cursor_callback(cur, batch)
                            if yield_result_rows:
                                for r in cur:
                                    yield r
                    batch = []

            if batch:
                self._log.debug('-- tail batch (size: %d)', len(batch))
                with conn:
                    with conn.cursor() as cur:
                        self.ensure_table(cur)
                    with conn.cursor(cursor_name) as cur:
                        cur.itersize = self.itersize
                        cursor_callback(cur, batch)
                        if yield_result_rows:
                            for r in cur:
                                yield r

            if conn is not None:
                conn.commit()
        except:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            # conn.__exit__ doesn't close connection, just the transaction
            if conn is not None:
                conn.close()
            self._log.debug('-- done')
