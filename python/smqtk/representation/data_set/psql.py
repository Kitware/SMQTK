import warnings

from smqtk.representation import DataSet
from smqtk.representation.data_element.psql import PostgresDataElement
from smqtk.utils.postgres import PsqlConnectionHelper

# Try to import required modules
try:
    import psycopg2
except ImportError:
    psycopg2 = None


class PostgresNativeDataSet (DataSet):
    """
    Dataset that stores data elements natively in a PostgreSQL database.

    Elements stored in this data set implementation will be copied as
    PostgresDataElements, which are stored based on this data set's
    configuration.

    Data elements retrieved from this data set will be of the
    PostgresDataElement class type.
    """

    @classmethod
    def is_usable(cls):
        if psycopg2 is None:
            warnings.warn("PostgresNativeDataSet not usable due to psycopg2 "
                          "package not being importable.")
            return False
        return True

    def __init__(self, table_name="psql_data_elements", id_col="id",
                 sha1_col="sha1", mime_col="mime", byte_col="bytes",
                 db_name="postgres", db_host="/tmp", db_port=5432,
                 db_user=None, db_pass=None, itersize=1000, read_only=False,
                 create_table=True):
        """
        Create a PostgreSQL-based data set instance.

        If the tabled mapped to the provided ``table_name`` already exists, we
        expect the provided columns to match the following types:
        - ``id_col`` is expected to be TEXT
        - ``sha1_col`` is expected to be TEXT
        - ``type_col`` is expected to be TEXT
        - ``byte_col`` is expected to be BYTEA

        Default database connection parameters are assuming the use of a
        non-default, non-postgres-user cluster where the current user's name is
        equivalent to a valid role in the database.

        :param str table_name:
            String label of the table in the database to interact with.
        :param str id_col:
            Name of the element ID column in ``table_name``.
        :param str sha1_col:
            Name of the SHA1 column in ``table_name``.
        :param str mime_col:
            Name of the MIMETYPE column in ``table_name``.
        :param str byte_col:
            Name of the column storing byte data in ``table_name``.
        :param str|None db_host:
            Host address of the PostgreSQL server. If None, we assume the
            server is on the local machine and use the UNIX socket. This might
            be a required field on Windows machines (not tested yet).
        :param str|None db_port:
            Port the Postgres server is exposed on. If None, we assume a
            default port (5433).
        :param str db_name:
            The name of the database to connect to.
        :param str|None db_user:
            Postgres user to connect as. If None, postgres defaults to using
            the current accessing user account name on the operating system.
        :param str|None db_pass:
            Password for the user we're connecting as. This may be None if no
            password is to be used.
        :param int itersize:
            Number of records fetched per network round trip when iterating
            over a named cursor.
        :param bool read_only:
            Only allow reading of this data.  Modification actions will throw a
            ReadOnlyError exceptions.
        :param bool create_table:
            If this instance should try to create the storing table before
            actions are performed against it. If the configured user does not
            have sufficient permissions to create the table and it does not
            currently exist, an exception will be raised.

        """
        super(PostgresNativeDataSet, self).__init__()

        itersize = int(itersize)
        if itersize <= 0:
            raise ValueError("Itersize must be greater than 0.")

        self._table_name = table_name
        self._id_col = id_col
        self._sha1_col = sha1_col
        self._mime_col = mime_col
        self._byte_col = byte_col
        self._read_only = read_only
        self._create_table = create_table

        self._psql_helper = PsqlConnectionHelper(db_name, db_host, db_port,
                                                 db_user, db_pass, itersize)

        # Set table creation SQL in helper
        if not self._read_only:
            self._psql_helper.set_table_upsert_sql(
                PostgresDataElement.CommandTemplates.UPSERT_TABLE.format(
                    table_name=self._table_name,
                    id_col=self._id_col,
                    sha1_col=self._sha1_col,
                    mime_col=self._mime_col,
                    byte_col=byte_col,
                )
            )

    def get_config(self):
        return {
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
            "itersize": self._psql_helper.itersize,

            "read_only": self._read_only,
            "create_table": self._create_table,
        }

    def _gen_psql_element(self, uid, content_type=None):
        """
        Internal method to generate a psql data element with appropriate psql
        parameters.
        :param collections.Hashable uid: UUID of data element.
        :param None|str content_type: Content type / MIME type of the element.
        """
        e = PostgresDataElement(
            uid, content_type=content_type, table_name=self._table_name,
            id_col=self._id_col, sha1_col=self._sha1_col,
            mime_col=self._mime_col, byte_col=self._byte_col,
            read_only=self._read_only, create_table=self._create_table
        )
        # Share PSQL helper instance.
        e._psql_helper = self._psql_helper
        return e

    def __iter__(self):
        """
        :return: Generator over the DataElements contained in this set in no
            particular order.
        """
        # Select all UUIDs and content type, yielding constructed psql
        # data elements.
        q = "SELECT {id_col:s}, {mime_col:s} FROM {table_name:s};".format(
            id_col=self._id_col,
            mime_col=self._mime_col,
            table_name=self._table_name,
        )

        def cb(cursor):
            cursor.execute(q)

        for r in self._psql_helper.single_execute(cb, yield_result_rows=True):
            e_uuid, e_mimetype = r
            yield self._gen_psql_element(e_uuid, e_mimetype)

    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        # Query count of rows in table (select on id col only)
        count_query = "SELECT count({id_col:s}) FROM {table_name:s};".format(
            id_col=self._id_col,
            table_name=self._table_name,
        )

        def cb(cursor):
            cursor.execute(count_query)

        r = list(self._psql_helper.single_execute(cb, yield_result_rows=True))
        if not r:
            # No rows in table
            return 0
        return int(r[0][0])

    def uuids(self):
        """
        :return: A new set of uuids represented in this data set.
        :rtype: set
        """
        # TODO: UPDATE TO ITERATOR INSTEAD OF SET RETURN TYPE
        q = "SELECT {id_col:s} FROM {table_name:s};".format(
            id_col=self._id_col,
            table_name=self._table_name,
        )

        def cb(cursor):
            cursor.execute(q)

        return {r[0] for r in
                self._psql_helper.single_execute(cb, yield_result_rows=True)}

    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the
            type that the set implementation expects or cares about.
        :type uuid: collections.Hashable

        :return: True if the given uuid matches an element in this set, or
            False if it does not.
        :rtype: bool

        """
        # Query for table id col values
        q = "SELECT {id_col:s} FROM {table_name:s} "\
            "WHERE {id_col:s} = %(id_val)s;"\
            .format(id_col=self._id_col,
                    table_name=self._table_name)

        def cb(cursor):
            cursor.execute(q, {'id_val': str(uuid)})

        return bool(list(
            self._psql_helper.single_execute(cb, yield_result_rows=True)
        ))

    def add_data(self, *elems):
        """
        Add the given data element(s) instance to this data set.

        *NOTE: Implementing methods should check that input elements are in
        fact DataElement instances.*

        :param elems: Data element(s) to add
        :type elems: smqtk.representation.DataElement

        """
        # TODO: Optimize for batch insertion using custom query.
        for e in elems:
            pe = self._gen_psql_element(e.uuid(), e.content_type())
            pe.set_bytes(e.get_bytes())

    def get_data(self, uuid):
        """
        Get the data element the given uuid references, or raise an
        exception if the uuid does not reference any element in this set.

        :raises KeyError: If the given uuid does not refer to an element in
            this data set.

        :param uuid: The uuid of the element to retrieve.
        :type uuid: collections.Hashable

        :return: The data element instance for the given uuid.
        :rtype: smqtk.representation.DataElement

        """
        # Query for content type recorded in our table to use for PSQL element
        # construction.
        q = "SELECT {ct_col:s} FROM {table_name:s} " \
            "WHERE {id_col:s} = %(id_val)s;" \
            .format(ct_col=self._mime_col,
                    table_name=self._table_name,
                    id_col=self._id_col)

        def cb(cursor):
            cursor.execute(q, {'id_val': str(uuid)})

        r = list(self._psql_helper.single_execute(cb, yield_result_rows=True))
        if not r:
            # No rows matching the input uuid were found.
            raise KeyError(uuid)

        # Create and return the PSQL element.
        ct = str(r[0][0])
        return self._gen_psql_element(uuid, content_type=ct)
