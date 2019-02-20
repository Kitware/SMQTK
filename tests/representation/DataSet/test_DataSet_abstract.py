from __future__ import division, print_function
import mock
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

        self.assertEqual(len(ds), expected_len)

    def test_getitem_mock(self):
        expected_key = 'foo'
        expected_value = 'bar'

        def expected_effect(k):
            if k == expected_key:
                return expected_value
            raise RuntimeError("not expected key")

        ds = DummyDataSet()
        ds.get_data = mock.MagicMock(side_effect=expected_effect)

        self.assertRaisesRegexp(
            RuntimeError,
            "^not expected key$",
            ds.__getitem__, 'unexpectedKey'
        )
        self.assertEqual(ds[expected_key], expected_value)

    def test_contains(self):
        """
        By mocking DummyDataSet's ``has_uuid`` method (an abstract method), we
        check that the ``__contains__`` method on the parent class does the
        right thing when using python's ``in`` syntax.
        """
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

        # noinspection PyTypeChecker
        # - Using a mock object on purpose in conjuction with ``has_uuid``
        #   override.
        self.assertTrue(mock_data_element in ds)
        ds.has_uuid.assert_called_once_with(expected_uuid)

        mock_data_element.uuid.return_value = 'not expected uuid'
        # noinspection PyTypeChecker
        # - Using a mock object on purpose in conjuction with ``has_uuid``
        #   override.
        self.assertFalse(mock_data_element in ds)
