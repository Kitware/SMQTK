import mock
from unittest import TestCase

import smqtk.exceptions
from smqtk.representation.data_element.hbase_element import HBaseDataElement
from smqtk.utils.configuration import configuration_test_helper


class TestHBaseDataElement(TestCase):

    DUMMY_CFG = {
        'element_key': 'foobar',
        'binary_column': 'binary_data',
        'hbase_address': 'some_address',
        'hbase_table': 'some_table',
        'timeout': 12345,
    }

    def make_element(self, content):
        """ Make a test HBaseDataElement based on the DUMMY_CFG that should
        return the given binary content. This sets up appropriate mocking for
        this instance.
        """
        e = HBaseDataElement(**self.DUMMY_CFG)
        # Pretend that the implementation is actually available and mock out
        # dependency functionality.
        e.content_type = mock.MagicMock()
        e._new_hbase_table_connection = mock.MagicMock()
        e._new_hbase_table_connection().row.return_value = {
            self.DUMMY_CFG['binary_column']: content
        }
        return e

    @classmethod
    def setUpClass(cls):
        # Pretend that the implementation is actually available and mock out
        # dependency functionality.
        HBaseDataElement.is_usable = mock.MagicMock(return_value=True)

    def test_config(self):
        inst = HBaseDataElement(
            element_key='foobar',
            binary_column='binary_data',
            hbase_address="some_address",
            hbase_table='some_table',
            timeout=12345
        )
        for i in configuration_test_helper(inst):  # type: HBaseDataElement
            assert i.element_key == 'foobar'
            assert i.binary_column == 'binary_data'
            assert i.hbase_address == 'some_address'
            assert i.hbase_table == 'some_table'
            assert i.timeout == 12345

    def test_get_bytes(self):
        expected_bytes = b'foo bar test string'
        e = self.make_element(expected_bytes)
        self.assertEqual(e.get_bytes(), expected_bytes)

    def test_is_empty_zero_bytes(self):
        # Simulate empty bytes
        e = self.make_element(b'')
        self.assertTrue(e.is_empty())

    def test_is_empty_nonzero_bytes(self):
        # Simulate non-empty bytes
        e = self.make_element(b'some bytes')
        self.assertFalse(e.is_empty())

    def test_writable(self):
        # Read-only element
        e = self.make_element(b'')
        self.assertFalse(e.writable())

    def test_set_bytes(self):
        # Read-only element
        e = self.make_element(b'')
        self.assertRaises(
            smqtk.exceptions.ReadOnlyError,
            e.set_bytes, b'some bytes'
        )
