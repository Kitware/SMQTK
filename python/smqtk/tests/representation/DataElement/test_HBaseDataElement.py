import mock
from unittest import TestCase

import nose.tools

import smqtk.exceptions
from smqtk.representation.data_element.hbase_element import HBaseDataElement


class TestHBaseDataElement(TestCase):

    DUMMY_CFG = {
        'element_key': 'foobar',
        'binary_column': 'binary_data',
        'hbase_address': 'some_address',
        'hbase_table': 'some_table',
        'timeout': 12345,
    }

    def make_element(self, content):
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
        cfg = HBaseDataElement.get_default_config()
        nose.tools.assert_equal(cfg, {
            'element_key': None,
            'binary_column': None,
            'hbase_address': None,
            'hbase_table': None,
            'timeout': 10000,
        })

        cfg = self.DUMMY_CFG
        e = HBaseDataElement.from_config(cfg)

        nose.tools.assert_equal(e.element_key, cfg['element_key'])
        nose.tools.assert_equal(e.binary_column, cfg['binary_column'])
        nose.tools.assert_equal(e.hbase_address, cfg['hbase_address'])
        nose.tools.assert_equal(e.hbase_table, cfg['hbase_table'])
        nose.tools.assert_equal(e.timeout, cfg['timeout'])

        # output should be the same as what we constructed with in this case
        e_get_cfg = e.get_config()
        nose.tools.assert_equal(e_get_cfg, cfg)

    def test_get_bytes(self):
        expected_bytes = 'foo bar test string'
        e = self.make_element(expected_bytes)
        nose.tools.assert_equal(e.get_bytes(), expected_bytes)

    def test_is_empty_zero_bytes(self):
        # Simulate empty bytes
        e = self.make_element('')
        nose.tools.assert_true(e.is_empty())

    def test_is_empty_nonzero_bytes(self):
        # Simulate non-empty bytes
        e = self.make_element('some bytes')
        nose.tools.assert_false(e.is_empty())

    def test_writable(self):
        # Read-only element
        e = self.make_element('')
        nose.tools.assert_false(e.writable())

    def test_set_bytes(self):
        # Read-only element
        e = self.make_element('')
        nose.tools.assert_raises(
            smqtk.exceptions.ReadOnlyError,
            e.set_bytes, 'some bytes'
        )
