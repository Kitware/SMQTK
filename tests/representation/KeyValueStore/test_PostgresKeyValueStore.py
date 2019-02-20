import mock
import unittest

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value.postgres import PostgresKeyValueStore


if PostgresKeyValueStore.is_usable():

    class TestPostgresKeyValueStore (unittest.TestCase):

        # noinspection PyUnusedLocal
        # - purposefully not used mock objects
        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove_readonly(self, m_gcp):
            """ Test that we cannot remove from readonly instance. """
            s = PostgresKeyValueStore(read_only=True)

            self.assertRaises(
                ReadOnlyError,
                s.remove, 0
            )

        # noinspection PyUnusedLocal
        # - purposefully not used mock objects
        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove_invalid_key(self, m_gcp):
            """
            Simulate an missing key and that it should result in a thrown
            KeyError
            """
            s = PostgresKeyValueStore()

            # Pretend this store contains nothing.
            s.has = mock.Mock(return_value=False)

            self.assertRaises(
                KeyError,
                s.remove, 0
            )

        # noinspection PyUnusedLocal
        # - purposefully not used mock objects
        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove(self, m_gcp):
            """
            Simulate removing a value from the store. Checking executions on
            the mock cursor.
            """
            expected_key = 'test_remove_key'
            expected_key_bytea = bytes(
                PostgresKeyValueStore._py_to_bin(expected_key))

            # Cut out create table calls.
            s = PostgresKeyValueStore(create_table=False)
            # Pretend key exists in index.
            s.has = mock.Mock(return_value=True)

            # Cursor is created via a context (i.e. __enter__()
            #: :type: mock.Mock
            mock_execute = s._psql_helper.get_psql_connection().cursor()\
                            .__enter__().execute

            s.remove(expected_key)

            # Call to ensure table and call to remove.
            mock_execute.assert_called_once()
            # Call should have been with provided key as converted to postgres
            # bytea type.
            self.assertRegexpMatches(mock_execute.call_args[0][0],
                                     "DELETE FROM .+ WHERE .+ LIKE .+")
            self.assertEqual(set(mock_execute.call_args[0][1].keys()),
                             {'key_like'})
            self.assertEqual(bytes(mock_execute.call_args[0][1]['key_like']),
                             expected_key_bytea)

        # noinspection PyUnusedLocal
        # - purposefully not used mock objects
        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove_many_readonly(self, m_gcp):
            """
            Test failure to remove from a readonly instance.
            """
            s = PostgresKeyValueStore(read_only=True)
            self.assertRaises(
                ReadOnlyError,
                s.remove_many, [0, 1]
            )

        # noinspection PyUnusedLocal
        # - purposefully not used mock objects
        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove_many_invalid_keys(self, m_gcp):
            """
            Test failure when one or more provided keys are not present in
            store.
            """
            s = PostgresKeyValueStore(create_table=False)

            # Simulate the batch execute returning nothing.  This simulates no
            # rows being found by the first call to the method when checking
            # for key presence in table.
            s._check_contained_keys = mock.Mock(return_value={0, 1})
            PY2_SET_KEY_ERROR_RE = "set\(\[(?:0|1), (?:0|1)\]\)"
            PY3_SET_KEY_ERROR_RE = "{(?:0|1), (?:0|1)}"
            self.assertRaisesRegexp(
                KeyError, '^(?:{}|{})$'.format(PY2_SET_KEY_ERROR_RE,
                                               PY3_SET_KEY_ERROR_RE),
                s.remove_many, [0, 1]
            )

            # Simulate only one of the keys existing in the table.
            s._check_contained_keys = mock.Mock(return_value={1})
            self.assertRaisesRegexp(
                KeyError, '^1$',
                s.remove_many, [0, 1]
            )
            s._check_contained_keys = mock.Mock(return_value={0})
            self.assertRaisesRegexp(
                KeyError, '^0$',
                s.remove_many, [0, 1]
            )

        # noinspection PyUnusedLocal
        # - purposefully not used mock objects
        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        # The underlying psycopg2 function used in callback provided to helper.
        @mock.patch('smqtk.representation.key_value.postgres.psycopg2.extras'
                    '.execute_batch')
        def test_remove_many(self, m_psqlExecBatch, m_gcp):
            """
            Test expected calls to psql `execute_batch` function when removing
            multiple items.
            """
            expected_key_1 = 'test_remove_many_key_1'
            exp_key_1_bytea = PostgresKeyValueStore._py_to_bin(expected_key_1)
            expected_key_2 = 'test_remove_many_key_2'
            exp_key_2_bytea = PostgresKeyValueStore._py_to_bin(expected_key_2)

            # Skip table creation calls for simplicity.
            s = PostgresKeyValueStore(create_table=False)

            # Mock PSQL cursor stuff because we aren't actually connecting to a
            # database.
            # - `get_psql_connection` uses `smqtk.utils.postgres.
            #   get_connection_pool`, so the return is a mock object.
            # - Cursor is created via a context (i.e. __enter__()) when utilized
            #   in `PsqlConnectionHelper` execute methods.
            #: :type: mock.Mock
            mock_cursor = s._psql_helper.get_psql_connection().cursor() \
                .__enter__()

            # Mocking `PostgresKeyValueStore` key-check method so as to pretend
            # that the given keys exist in the database
            s._check_contained_keys = mock.MagicMock(return_value=set())

            s.remove_many([expected_key_1, expected_key_2])

            # As a result of this call, we expect:
            # - ``psycopg2.extras.execute_batch`` should have been called once
            #   when deleting key-value pairs in db (2 < `s._batch_size`)
            #
            # We back to break up the argument equality check of recorded mock
            # function call arguments due to ``psycopg2.Binary`` instances not
            # being comparable.

            m_psqlExecBatch.assert_called_once()

            # Confirm call arguments provided to
            # ``psycopg2.extras.execute_batch`` are as expected.
            # -
            expected_del_q = "DELETE FROM data_set WHERE key LIKE %(key_like)s"
            psqlExecBatch_call_args = m_psqlExecBatch.call_args[0]
            psqlExecBatch_kwargs = m_psqlExecBatch.call_args[1]
            self.assertEqual(psqlExecBatch_call_args[0], mock_cursor)
            self.assertEqual(psqlExecBatch_call_args[1], expected_del_q)
            # 3rd argument is a list of dictionaries for 'key_like' replacements
            # - dictionary values are `psycopg2.extensions.Binary` type, which
            #   are not directly comparable (different instances). Have to
            #   convert to bytes representation in order to compare.
            self.assertEqual(len(psqlExecBatch_call_args[2]), 2)
            self.assertListEqual(
                [bytes(d['key_like']) for d in psqlExecBatch_call_args[2]],
                [bytes(exp_key_1_bytea), bytes(exp_key_2_bytea)]
            )
            self.assertIn('page_size', psqlExecBatch_kwargs)
            self.assertEqual(psqlExecBatch_kwargs['page_size'], s._batch_size)
            self.assertEqual(psqlExecBatch_kwargs['page_size'], 1000)  # default
