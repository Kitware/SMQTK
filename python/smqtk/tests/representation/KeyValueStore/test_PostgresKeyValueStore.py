import mock
import unittest

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value.postgres import PostgresKeyValueStore


if PostgresKeyValueStore.is_usable():

    class TestPostgresKeyValueStore (unittest.TestCase):

        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove_readonly(self, m_gcp):
            """ Test that we cannot remove from readonly instance. """
            s = PostgresKeyValueStore(read_only=True)

            self.assertRaises(
                ReadOnlyError,
                s.remove, 0
            )

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

        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove(self, m_gcp):
            """
            Simulate removing a value from the store. Checking executions on
            the mock cursor.
            """
            # Cut out create table calls.
            s = PostgresKeyValueStore(create_table=False)
            # Pretend key exists in index.
            s.has = mock.Mock(return_value=True)

            # Cursor is created via a context (i.e. __enter__()
            #: :type: mock.Mock
            mock_execute = s._psql_helper.get_psql_connection().cursor()\
                            .__enter__().execute

            expected_key = 'test_remove_key'
            s.remove(expected_key)

            # Call to ensure table and call to remove.
            mock_execute.assert_called_once()
            # Call should have been with provided key.
            self.assertRegexpMatches(mock_execute.call_args[0][0],
                                     "DELETE FROM .+ WHERE .+ LIKE .+")
            self.assertDictEqual(mock_execute.call_args[0][1],
                                 {'key_like': expected_key})

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
            s._psql_helper.batch_execute = mock.Mock(return_value=[])
            PY2_SET_KEY_ERROR_RE = "set\(\[(?:0|1), (?:0|1)\]\)"
            PY3_SET_KEY_ERROR_RE = "{(?:0|1), (?:0|1)}"
            self.assertRaisesRegexp(
                KeyError, '^(?:{}|{})$'.format(PY2_SET_KEY_ERROR_RE,
                                               PY3_SET_KEY_ERROR_RE),
                s.remove_many, [0, 1]
            )

            # Simulate only one of the keys existing in the table.
            s._psql_helper.batch_execute = mock.Mock(return_value=[[0]])
            self.assertRaisesRegexp(
                KeyError, '^1$',
                s.remove_many, [0, 1]
            )
            s._psql_helper.batch_execute = mock.Mock(return_value=[[1]])
            self.assertRaisesRegexp(
                KeyError, '^0$',
                s.remove_many, [0, 1]
            )

        @mock.patch('smqtk.utils.postgres.get_connection_pool')
        def test_remove_many(self, m_gcp):
            """
            Test expected calls to psql cursor during normal operation.
            """
            expected_key_1 = 'test_remove_many_key_1'
            expected_key_2 = 'test_remove_many_key_2'

            # Skip table creation calls.
            s = PostgresKeyValueStore(create_table=False)

            # Cursor is created via a context (i.e. __enter__()
            #: :type: mock.Mock
            mock_cursor = s._psql_helper.get_psql_connection().cursor() \
                .__enter__()
            # Make the cursor iterate keys in order to pass key check.
            mock_cursor.__iter__.return_value = [[expected_key_1],
                                                 [expected_key_2]]
            #: :type: mock.Mock
            mock_execute = mock_cursor.execute

            s.remove_many([expected_key_1, expected_key_2])

            # Cursor should have been executed 2 times, 1 for batch key check
            # and 1 for batch deletion.  1 call for each since 2 < default
            # batch size of 1000.
            self.assertEqual(2, mock_execute.call_count)

            expected_has_q = "SELECT key FROM data_set WHERE key LIKE " \
                             "%(key_like)s"
            expected_has_vals = [{'key_like': expected_key_1},
                                 {'key_like': expected_key_2}]
            mock_execute.assert_any_call(expected_has_q, expected_has_vals)

            expected_del_q = "DELETE FROM data_set WHERE key LIKE %(key_like)s"
            expected_del_vals = [{'key_like': expected_key_1},
                                 {'key_like': expected_key_2}]
            mock_execute.assert_any_call(expected_del_q, expected_del_vals)
