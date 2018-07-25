import mock
import unittest

from smqtk.exceptions import ReadOnlyError
from smqtk.representation.key_value.postgres import PostgresKeyValueStore


if PostgresKeyValueStore.is_usable():

    class TestPostgresKeyValueStore (unittest.TestCase):

        @staticmethod
        def _mock_psql_helper(s):
            """
            :type s: PostgresKeyValueStore
            """
            s._psql_helper.get_psql_connection = mock.MagicMock()

        def test_remove_readonly(self):
            """ Test that we cannot remove from readonly instance. """
            s = PostgresKeyValueStore(read_only=True)
            self._mock_psql_helper(s)

            self.assertRaises(
                ReadOnlyError,
                s.remove, 0
            )

        def test_remove_invalid_key(self):
            """
            Simulate an missing key and that it should result in a thrown
            KeyError
            """
            s = PostgresKeyValueStore()
            self._mock_psql_helper(s)

            # Pretend this store contains nothing.|
            s.has = mock.Mock(return_value=False)

            self.assertRaises(
                KeyError,
                s.remove, 0
            )

        def test_remove(self):
            """
            Simulate removing a value from the store. Checking executions on
            the mock cursor.
            """
            # Cut out create table calls.
            s = PostgresKeyValueStore(create_table=False)
            self._mock_psql_helper(s)
            # Pretend key exists in index.
            s.has = mock.Mock(return_value=True)

            # Cursor is created via a context (i.e. __enter__()
            #: :type: mock.Mock
            mock_execute = s._psql_helper.get_psql_connection().cursor()\
                            .__enter__().execute

            s.remove(0)

            # Call to ensure table and call to remove.
            mock_execute.assert_called_once()
            # Call should have been with provided key.
            self.assertRegexpMatches(mock_execute.call_args[0][0],
                                     "DELETE FROM .+ WHERE .+ LIKE .+")
            self.assertDictEqual(mock_execute.call_args[0][1],
                                 {'key_like': 0})

        def test_remove_many_readonly(self):
            """
            Test failure to remove from a readonly instance.
            """
            s = PostgresKeyValueStore(read_only=True)
            self._mock_psql_helper(s)
            self.assertRaises(
                ReadOnlyError,
                s.remove_many, [0, 1]
            )

        def test_remove_many_invalid_keys(self):
            """
            Test failure when one or more provided keys are not present in
            store.
            """
            s = PostgresKeyValueStore(create_table=False)
            self._mock_psql_helper(s)

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

        def test_remove_many(self):
            """
            Test expected calls to psql cursor during normal operation.
            """
            # Skip table creation calls.
            s = PostgresKeyValueStore(create_table=False)
            self._mock_psql_helper(s)

            # Cursor is created via a context (i.e. __enter__()
            #: :type: mock.Mock
            mock_cursor = s._psql_helper.get_psql_connection().cursor() \
                .__enter__()
            # Make the cursor iterate keys in order to pass key check.
            mock_cursor.__iter__.return_value = [[0], [1]]
            #: :type: mock.Mock
            mock_execute = mock_cursor.execute

            s.remove_many([0, 1])

            # Cursor should have been executed 2 times, 1 for batch key check
            # and 1 for batch deletion.  1 call for each since 2 < default
            # batch size of 1000.
            self.assertEqual(mock_execute.call_count, 2)

            expected_has_q = "SELECT key FROM data_set WHERE key LIKE " \
                             "%(key_like)s"
            expected_has_vals = [{'key_like': 0}, {'key_like': 1}]
            mock_execute.assert_any_call(expected_has_q, expected_has_vals)

            expected_del_q = "DELETE FROM data_set WHERE key LIKE %(key_like)s"
            expected_del_vals = [{'key_like': 0}, {'key_like': 1}]
            mock_execute.assert_any_call(expected_del_q, expected_del_vals)
