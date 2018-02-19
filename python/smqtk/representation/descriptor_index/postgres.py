"""

References:
    [1] http://stackoverflow.com/questions/866465/sql-order-by-the-in-value-list
        answer from "a_horse_with_no_name"

"""
import itertools
import logging
import multiprocessing
import uuid

try:
    import cPickle as pickle
except ImportError:
    import pickle

from smqtk.representation import DescriptorIndex
from smqtk.exceptions import ReadOnlyError
from smqtk.utils.postgres \
    import norm_psql_cmd_string, PsqlConnectionHelper

try:
    import psycopg2
except ImportError as ex:
    logging.getLogger(__name__)\
           .warning("Failed to import psycopg2: %s", str(ex))
    psycopg2 = None


PSQL_TABLE_CREATE_RLOCK = multiprocessing.RLock()


# noinspection SqlNoDataSourceInspection
class PostgresDescriptorIndex (DescriptorIndex):
    """
    DescriptorIndex implementation that stored DescriptorElement references in
    a PostgreSQL database.

    A ``PostgresDescriptorIndex`` effectively controls the entire table. Thus
    a ``clear()`` call will remove everything from the table.

    PostgreSQL version support:
        - 9.4

    Table format:
        <uuid col>      TEXT NOT NULL
        <element col>   BYTEA NOT NULL

        <uuid_col> should be the primary key (we assume unique).

    We require that the no column labels not be 'true' for the use of a value
    return shortcut.

    """

    #
    # The following are SQL query templates. The string formatting using {}'s
    # is used to fill in the query before using it in an execute with instance
    # specific values. The ``%()s`` formatting is special for the execute where-
    # by psycopg2 will fill in the values appropriately as specified in a second
    # dictionary argument to ``cursor.execute(query, value_dict)``.
    #
    UPSERT_TABLE_TMPL = norm_psql_cmd_string("""
        CREATE TABLE IF NOT EXISTS {table_name:s} (
          {uuid_col:s} TEXT NOT NULL,
          {element_col:s} BYTEA NOT NULL,
          PRIMARY KEY ({uuid_col:s})
        );
    """)

    SELECT_TMPL = norm_psql_cmd_string("""
        SELECT {col:s}
          FROM {table_name:s}
    """)

    SELECT_LIKE_TMPL = norm_psql_cmd_string("""
        SELECT {element_col:s}
          FROM {table_name:s}
         WHERE {uuid_col:s} like %(uuid_like)s
    """)

    # So we can ensure we get back elements in specified order
    #   - reference [1]
    SELECT_MANY_ORDERED_TMPL = norm_psql_cmd_string("""
        SELECT {table_name:s}.{element_col:s}
          FROM {table_name:s}
          JOIN (
            SELECT *
            FROM unnest(%(uuid_list)s) with ordinality
          ) AS __ordering__ ({uuid_col:s}, {uuid_col:s}_order)
            ON {table_name:s}.{uuid_col:s} = __ordering__.{uuid_col:s}
          ORDER BY __ordering__.{uuid_col:s}_order
    """)

    UPSERT_TMPL = norm_psql_cmd_string("""
        WITH upsert AS (
          UPDATE {table_name:s}
            SET {element_col:s} = %(element_val)s
            WHERE {uuid_col:s} = %(uuid_val)s
            RETURNING *
          )
        INSERT INTO {table_name:s}
          ({uuid_col:s}, {element_col:s})
          SELECT %(uuid_val)s, %(element_val)s
            WHERE NOT EXISTS (SELECT * FROM upsert)
    """)

    DELETE_LIKE_TMPL = norm_psql_cmd_string("""
        DELETE FROM {table_name:s}
              WHERE {uuid_col:s} like %(uuid_like)s
    """)

    DELETE_MANY_TMPL = norm_psql_cmd_string("""
        DELETE FROM {table_name:s}
              WHERE {uuid_col:s} in %(uuid_tuple)s
          RETURNING uid
    """)

    @classmethod
    def is_usable(cls):
        return psycopg2 is not None

    def __init__(self, table_name='descriptor_index', uuid_col='uid',
                 element_col='element',
                 db_name='postgres', db_host=None, db_port=None, db_user=None,
                 db_pass=None, multiquery_batch_size=1000, pickle_protocol=-1,
                 read_only=False, create_table=True):
        """
        Initialize index instance.

        :param table_name: Name of the table to use.
        :type table_name: str

        :param uuid_col: Name of the column containing the UUID signatures.
        :type uuid_col: str

        :param element_col: Name of the table column that will contain
            serialized elements.
        :type element_col: str

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

        :param multiquery_batch_size: For queries that handle sending or
            receiving many queries at a time, batch queries based on this size.
            If this is None, then no batching occurs.

            The advantage of batching is that it reduces the memory impact for
            queries dealing with a very large number of elements (don't have to
            store the full query for all elements in RAM), but the transaction
            will be some amount slower due to splitting the query into multiple
            transactions.
        :type multiquery_batch_size: int | None

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
        super(PostgresDescriptorIndex, self).__init__()

        self.table_name = table_name
        self.uuid_col = uuid_col
        self.element_col = element_col

        self.multiquery_batch_size = multiquery_batch_size
        self.pickle_protocol = pickle_protocol
        self.read_only = bool(read_only)
        self.create_table = create_table

        # Checking parameters where necessary
        if self.multiquery_batch_size is not None:
            self.multiquery_batch_size = int(self.multiquery_batch_size)
            assert self.multiquery_batch_size > 0, \
                "A given batch size must be greater than 0 in size " \
                "(given: %d)." % self.multiquery_batch_size
        assert -1 <= self.pickle_protocol <= 2, \
            ("Given pickle protocol is not in the known valid range. Given: %s"
             % self.pickle_protocol)

        self.psql_helper = PsqlConnectionHelper(db_name, db_host, db_port,
                                                db_user, db_pass,
                                                self.multiquery_batch_size,
                                                PSQL_TABLE_CREATE_RLOCK)
        if not self.read_only:
            self.psql_helper.set_table_upsert_sql(
                self.UPSERT_TABLE_TMPL.format(
                    table_name=self.table_name,
                    uuid_col=self.uuid_col,
                    element_col=self.element_col,
                )
            )

    def get_config(self):
        return {
            "table_name": self.table_name,
            "uuid_col": self.uuid_col,
            "element_col": self.element_col,

            "db_name": self.psql_helper.db_name,
            "db_host": self.psql_helper.db_host,
            "db_port": self.psql_helper.db_port,
            "db_user": self.psql_helper.db_user,
            "db_pass": self.psql_helper.db_pass,

            "multiquery_batch_size": self.multiquery_batch_size,
            "pickle_protocol": self.pickle_protocol,
            "read_only": self.read_only,
            "create_table": self.create_table,
        }

    def count(self):
        """
        :return: Number of descriptor elements stored in this index.
        :rtype: int | long
        """
        # Just count UUID column to limit data read.
        q = self.SELECT_TMPL.format(
            col='count(%s)' % self.uuid_col,
            table_name=self.table_name,
        )

        def exec_hook(cur):
            cur.execute(q)

        # There's only going to be one row returned with one element in it
        return list(self.psql_helper.single_execute(exec_hook, True))[0][0]

    def clear(self):
        """
        Clear this descriptor index's entries.
        """
        if self.read_only:
            raise ReadOnlyError("Cannot clear a read-only index.")

        q = self.DELETE_LIKE_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )

        def exec_hook(cur):
            cur.execute(q, {'uuid_like': '%'})

        list(self.psql_helper.single_execute(exec_hook))

    def has_descriptor(self, uuid):
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for
        :type uuid: collections.Hashable

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.
        :rtype: bool

        """
        q = self.SELECT_LIKE_TMPL.format(
            # hacking return value to something simple
            element_col='true',
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )

        def exec_hook(cur):
            cur.execute(q, {'uuid_like': str(uuid)})

        # Should either yield one or zero rows
        return bool(list(self.psql_helper.single_execute(exec_hook, True)))

    def add_descriptor(self, descriptor):
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptor: Descriptor to index.
        :type descriptor: smqtk.representation.DescriptorElement

        """
        if self.read_only:
            raise ReadOnlyError("Cannot clear a read-only index.")

        q = self.UPSERT_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
            element_col=self.element_col,
        )
        v = {
            'uuid_val': str(descriptor.uuid()),
            'element_val': psycopg2.Binary(
                pickle.dumps(descriptor, self.pickle_protocol)
            )
        }

        def exec_hook(cur):
            cur.execute(q, v)

        list(self.psql_helper.single_execute(exec_hook))

    def add_many_descriptors(self, descriptors):
        """
        Add multiple descriptors at one time.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptors: Iterable of descriptor instances to add to this
            index.
        :type descriptors:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        if self.read_only:
            raise ReadOnlyError("Cannot clear a read-only index.")

        q = self.UPSERT_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
            element_col=self.element_col,
        )

        # Transform input into
        def iter_elements():
            for d in descriptors:
                yield {
                    'uuid_val': str(d.uuid()),
                    'element_val': psycopg2.Binary(
                        pickle.dumps(d, self.pickle_protocol)
                    )
                }

        def exec_hook(cur, batch):
            cur.executemany(q, batch)

        self._log.debug("Adding many descriptors")
        list(self.psql_helper.batch_execute(iter_elements(), exec_hook,
                                            self.multiquery_batch_size))

    def get_descriptor(self, uuid):
        """
        Get the descriptor in this index that is associated with the given UUID.

        :param uuid: UUID of the DescriptorElement to get.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        :return: DescriptorElement associated with the queried UUID.
        :rtype: smqtk.representation.DescriptorElement

        """
        q = self.SELECT_LIKE_TMPL.format(
            element_col=self.element_col,
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )
        v = {'uuid_like': str(uuid)}

        def eh(c):
            c.execute(q, v)
            if c.rowcount == 0:
                raise KeyError(uuid)
            elif c.rowcount != 1:
                raise RuntimeError("Found more than one entry for the given "
                                   "uuid '%s' (got: %d)"
                                   % (uuid, c.rowcount))

        r = list(self.psql_helper.single_execute(eh, True))
        return pickle.loads(str(r[0][0]))

    def get_many_descriptors(self, uuids):
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated to given uuid values.
        :rtype: __generator[smqtk.representation.DescriptorElement]

        """
        q = self.SELECT_MANY_ORDERED_TMPL.format(
            table_name=self.table_name,
            element_col=self.element_col,
            uuid_col=self.uuid_col,
        )

        # Cache UUIDs received in order so we can check when we miss one in
        # order to raise a KeyError.
        uuid_order = []

        def iterelems():
            for uid in uuids:
                uuid_order.append(uid)
                yield str(uid)

        def exec_hook(cur, batch):
            v = {'uuid_list': batch}
            # self._log.debug('query: %s', cur.mogrify(q, v))
            cur.execute(q, v)

        self._log.debug("Getting many descriptors")
        # The SELECT_MANY_ORDERED_TMPL query ensures that elements returned are
        #   in the UUID order given to this method. Thus, if the iterated UUIDs
        #   and iterated return rows do not exactly line up, the query join
        #   failed to match a query UUID to something in the database.
        #   - We also check that the number of rows we got back is the same
        #     as elements yielded, else there were trailing UUIDs that did not
        #     match anything in the database.
        g = self.psql_helper.batch_execute(iterelems(), exec_hook,
                                           self.multiquery_batch_size, True)
        i = 0
        for r, expected_uuid in itertools.izip(g, uuid_order):
            d = pickle.loads(str(r[0]))
            if d.uuid() != expected_uuid:
                raise KeyError(expected_uuid)
            yield d
            i += 1

        if len(uuid_order) != i:
            # just report the first one that's bad
            raise KeyError(uuid_order[i])

    def remove_descriptor(self, uuid):
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.
        :type uuid: collections.Hashable

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        """
        if self.read_only:
            raise ReadOnlyError("Cannot remove from a read-only index.")

        q = self.DELETE_LIKE_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )
        v = {'uuid_like': str(uuid)}

        def execute(c):
            c.execute(q, v)
            # Nothing deleted if rowcount == 0
            # (otherwise 1 when deleted a thing)
            if c.rowcount == 0:
                raise KeyError(uuid)

        list(self.psql_helper.single_execute(execute))

    def remove_many_descriptors(self, uuids):
        """
        Remove descriptors associated to given descriptor UUIDs from this index.

        :param uuids: Iterable of descriptor UUIDs to remove.
        :type uuids: collections.Iterable[collections.Hashable]

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """
        if self.read_only:
            raise ReadOnlyError("Cannot remove from a read-only index.")

        q = self.DELETE_MANY_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )
        str_uuid_set = set(str(uid) for uid in uuids)
        v = {'uuid_tuple': tuple(str_uuid_set)}

        def execute(c):
            c.execute(q, v)

            # Check query UUIDs against rows that would actually be deleted.
            deleted_uuid_set = set(r[0] for r in c.fetchall())
            for uid in str_uuid_set:
                if uid not in deleted_uuid_set:
                    raise KeyError(uid)

        list(self.psql_helper.single_execute(execute))

    def iterkeys(self):
        """
        Return an iterator over indexed descriptor keys, which are their UUIDs.
        :rtype: collections.Iterator[collections.Hashable]
        """
        # Getting UUID through the element because the UUID might not be a
        # string type, and the true type is encoded with the DescriptorElement
        # instance.
        for d in self.iterdescriptors():
            yield d.uuid()

    def iterdescriptors(self):
        """
        Return an iterator over indexed descriptor element instances.
        :rtype: collections.Iterator[smqtk.representation.DescriptorElement]
        """
        def execute(c):
            c.execute(self.SELECT_TMPL.format(
                col=self.element_col,
                table_name=self.table_name
            ))

        #: :type: __generator
        execution_results = self.psql_helper.single_execute(execute, True, named=True)
        for r in execution_results:
            d = pickle.loads(str(r[0]))
            yield d

    def iteritems(self):
        """
        Return an iterator over indexed descriptor key and instance pairs.
        :rtype: collections.Iterator[(collections.Hashable,
                                      smqtk.representation.DescriptorElement)]
        """
        for d in self.iterdescriptors():
            yield d.uuid(), d
