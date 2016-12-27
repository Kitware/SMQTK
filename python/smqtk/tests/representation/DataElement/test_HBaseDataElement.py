import mock
from unittest import TestCase

import nose.tools

import smqtk.exceptions
from smqtk.representation.data_element.hbase_element import HBaseDataElement


if HBaseDataElement.is_usable():

    class TestHBaseDataElement(TestCase):

        DUMMY_CFG = {
            'element_key': 'foobar',
            'binary_column': 'binary_data',
            'hbase_address': 'some_address',
            'hbase_table': 'some_table',
            'timeout': 12345,
        }

        def test_config(self):
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

        def test_is_empty_zero_bytes(self):
            e = HBaseDataElement(**self.DUMMY_CFG)
            # Simulate empty bytes
            e.get_bytes = mock.MagicMock(return_value='')
            nose.tools.assert_true(e.is_empty())

        def test_is_empty_nonzero_bytes(self):
            e = HBaseDataElement(**self.DUMMY_CFG)
            # Simulate non-empty bytes
            e.get_bytes = mock.MagicMock(return_value='some bytes')
            nose.tools.assert_false(e.is_empty())

        def test_writable(self):
            # Read-only element
            e = HBaseDataElement(**self.DUMMY_CFG)
            nose.tools.assert_false(e.writable())

        def test_set_bytes(self):
            # Read-only element
            e = HBaseDataElement(**self.DUMMY_CFG)
            nose.tools.assert_raises(
                smqtk.exceptions.ReadOnlyError,
                e.set_bytes, 'some bytes'
            )
