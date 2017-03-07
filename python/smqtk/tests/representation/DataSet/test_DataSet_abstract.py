import mock
import nose.tools as ntools
import random
import unittest

from smqtk.representation import DataSet


class DummyDataSet (DataSet):

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self):
        super(DummyDataSet, self).__init__()

    def __iter__(self):
        pass

    def count(self):
        pass

    def uuids(self):
        pass

    def has_uuid(self, uuid):
        pass

    def add_data(self, *elems):
        pass

    def get_data(self, uuid):
        pass

    def get_config(self):
        return {}


class TestDataSetAbstract (unittest.TestCase):

    def test_len(self):
        expected_len = 134623456

        ds = DummyDataSet()
        ds.count = mock.MagicMock(return_value=expected_len)

        ntools.assert_equal(len(ds), expected_len)

    def test_getitem_mock(self):
        expected_key = 'foo'
        expected_value = 'bar'

        def expected_effect(k):
            if k == expected_key:
                return expected_value
            raise RuntimeError("not expected key")

        ds = DummyDataSet()
        ds.get_data = mock.MagicMock(side_effect=expected_effect)

        ntools.assert_raises_regexp(
            RuntimeError,
            "^not expected key$",
            ds.__getitem__, 'unexpectedKey'
        )
        ntools.assert_equal(ds[expected_key], expected_value)

    def test_contains(self):
        # Contains built-in hook expects data element and requests UUID from
        # that.
        expected_uuid = 'some uuid'

        mock_data_element = mock.MagicMock()
        mock_data_element.uuid = mock.MagicMock(return_value=expected_uuid)

        def expected_has_uuid_effect(k):
            if k == expected_uuid:
                return True
            return False

        ds = DummyDataSet()
        ds.has_uuid = mock.MagicMock(side_effect=expected_has_uuid_effect)

        ntools.assert_true(mock_data_element in ds)
        ds.has_uuid.assert_called_once_with(expected_uuid)

        mock_data_element.uuid.return_value = 'not expected uuid'
        ntools.assert_false(mock_data_element in ds)
