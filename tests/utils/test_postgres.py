import unittest

import pytest

from smqtk.utils.postgres import psycopg2, PsqlConnectionHelper


@pytest.mark.skipif(psycopg2 is None,
                    reason="Psycopg2 module is not importable, postgres "
                           "utilities are not available.")
class TestPsqlConnectionHelper (unittest.TestCase):

    def setUp(self):
        self.conn_helper = PsqlConnectionHelper()

    def test_batch_execute_on_empty_iterable(self):
        # noinspection PyUnusedLocal
        def exec_hook(cur, batch):
            raise Exception('This line shouldn\'t be reached with an empty '
                            'iterable.')

        list(self.conn_helper.batch_execute(iter(()), exec_hook, 1))
