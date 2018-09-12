from __future__ import division, print_function
import unittest

import mock
import numpy
import six

from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.algorithms.nn_index import NearestNeighborsIndex, get_nn_index_impls
from smqtk.utils.iter_validation import check_empty_iterable


class DummySI (NearestNeighborsIndex):

    @classmethod
    def is_usable(cls):
        """ stub """
        return True

    def get_config(self):
        """ stub """

    def _build_index(self, descriptors):
        """ stub """

    def _update_index(self, descriptors):
        """ stub """

    def _remove_from_index(self, uids):
        """ stub """

    def _nn(self, d, n=1):
        """ stub """

    def count(self):
        return 0


class TestNNIndexAbstract (unittest.TestCase):

    def test_get_impls(self):
        # Some implementations should be returned
        m = get_nn_index_impls()
        self.assertTrue(m)
        for cls in six.itervalues(m):
            self.assertTrue(issubclass(cls, NearestNeighborsIndex))

    def test_empty_iterable_exception(self):
        v = DummySI._empty_iterable_exception()
        self.assertIsInstance(v, ValueError)
        self.assertRegexpMatches(str(v), "DescriptorElement")

    def test_check_empty_iterable_no_data(self):
        # Test that an exception is thrown when an empty list/iterable is
        # passed.  Additionally check that the exception thrown has expected
        # message from exception generation method.
        callback = mock.MagicMock()

        # Not-stateful iterable (list)
        self.assertRaisesRegexp(
            ValueError,
            str(DummySI._empty_iterable_exception()),
            check_empty_iterable, [], callback,
            DummySI._empty_iterable_exception()
        )
        callback.assert_not_called()

        # with a stateful iterator.
        self.assertRaisesRegexp(
            ValueError,
            str(DummySI._empty_iterable_exception()),
            check_empty_iterable, iter([]), callback,
            DummySI._empty_iterable_exception()
        )
        callback.assert_not_called()

    def test_check_empty_iterable_valid_iterable(self):
        # Test that the method correctly calls the callback with the full
        # iterable when what is passed is not empty.
        callback = mock.MagicMock()

        # non-stateful iterator (set)
        d_set = {0, 1, 2, 3, 4}
        check_empty_iterable(d_set, callback,
                             DummySI()._empty_iterable_exception())
        callback.assert_called_once()
        self.assertSetEqual(
            set(callback.call_args[0][0]),
            d_set
        )

        # Stateful iterator
        callback = mock.MagicMock()
        check_empty_iterable(iter(d_set), callback,
                             DummySI()._empty_iterable_exception())
        callback.assert_called_once()
        self.assertSetEqual(
            set(callback.call_args[0][0]),
            d_set
        )

    def test_count_and_len(self):
        index = DummySI()
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.count(), len(index))

        # Pretend that there were things in there. Len should pass it though
        index.count = mock.Mock()
        index.count.return_value = 5
        self.assertEqual(len(index), 5)

    def test_build_index_no_descriptors(self):
        index = DummySI()
        index._build_index = mock.MagicMock()
        self.assertRaises(
            ValueError,
            index.build_index,
            []
        )
        index._build_index.assert_not_called()

    def test_build_index_nonzero_descriptors(self):
        index = DummySI()
        index._build_index = mock.MagicMock()
        d = DescriptorMemoryElement('test', 0)
        index.build_index([d])
        index._build_index.assert_called_once()
        # Check that the last call's first (only) argument was the same iterable
        # given.
        self.assertSetEqual(
            set(index._build_index.call_args[0][0]),
            {d}
        )

    def test_build_index_iterable(self):
        # Test build check with a pure iterable
        index = DummySI()
        index._build_index = mock.MagicMock()
        d_set = {
            DescriptorMemoryElement('test', 0),
            DescriptorMemoryElement('test', 1),
            DescriptorMemoryElement('test', 2),
            DescriptorMemoryElement('test', 3),
        }
        it = iter(d_set)
        index.build_index(it)
        # _build_index should have been called and the contents of the iterable
        # it was called with should equal d_set.
        index._build_index.assert_called_once()
        self.assertSetEqual(
            set(index._build_index.call_args[0][0]),
            d_set
        )

    def test_update_index_no_descriptors(self):
        index = DummySI()
        index._update_index = mock.MagicMock()
        self.assertRaises(
            ValueError,
            index.update_index,
            []
        )
        # internal method should not have been called.
        index._update_index.assert_not_called()

    def test_update_index_nonzero_descriptors(self):
        index = DummySI()
        index._update_index = mock.MagicMock()

        # Testing with dummy input data.
        d_set = {
            DescriptorMemoryElement('test', 0),
            DescriptorMemoryElement('test', 1),
            DescriptorMemoryElement('test', 2),
            DescriptorMemoryElement('test', 3),
        }
        index.update_index(d_set)
        index._update_index.assert_called_once()
        self.assertSetEqual(
            set(index._update_index.call_args[0][0]),
            d_set
        )

    def test_update_index_iterable(self):
        # Test build check with a pure iterable.
        index = DummySI()
        index._update_index = mock.MagicMock()
        d_set = {
            DescriptorMemoryElement('test', 0),
            DescriptorMemoryElement('test', 1),
            DescriptorMemoryElement('test', 2),
            DescriptorMemoryElement('test', 3),
        }
        it = iter(d_set)
        index.update_index(it)

        index._update_index.assert_called_once()
        self.assertSetEqual(
            set(index._update_index.call_args[0][0]),
            d_set,
        )

    def test_remove_from_index_no_uids(self):
        # Test that the method errors when no UIDs are provided
        index = DummySI()
        index._remove_from_index = mock.Mock()
        self.assertRaises(
            ValueError,
            index.remove_from_index, []
        )
        index._remove_from_index.assert_not_called()

    def test_remove_from_index_nonzero_descriptors(self):
        # Test removing a non-zero amount of descriptors
        index = DummySI()
        index._remove_from_index = mock.MagicMock()

        # Testing with dummy input data.
        uid_set = {0, 1, 2, 3}
        index.remove_from_index(uid_set)
        index._remove_from_index.assert_called_once()
        self.assertSetEqual(
            set(index._remove_from_index.call_args[0][0]),
            uid_set
        )

    def test_remove_from_index_nonzero_iterable(self):
        # Test removing a non-zero amount of descriptors via an iterable.
        index = DummySI()
        index._remove_from_index = mock.MagicMock()
        d_set = {0, 1, 2, 3}
        it = iter(d_set)
        index.remove_from_index(it)

        index._remove_from_index.assert_called_once()
        self.assertSetEqual(
            set(index._remove_from_index.call_args[0][0]),
            d_set,
        )

    def test_nn_empty_vector(self):
        # ValueError should be thrown if the input element has no vector.
        index = DummySI()
        # Need to force a non-zero index size for knn to be performed.
        index.count = mock.MagicMock(return_value=1)
        # Observe internal function
        index._nn = mock.MagicMock()

        q = DescriptorMemoryElement('test', 0)
        self.assertRaises(
            ValueError,
            index.nn, q
        )
        # template method should not have been called.
        index._nn.assert_not_called()

    def test_nn_empty_index(self):
        # nn should fail if index size is 0
        index = DummySI()
        index.count = mock.MagicMock(return_value=0)
        index._nn = mock.MagicMock()

        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        self.assertRaises(
            ValueError,
            index.nn, q
        )

    # noinspection PyUnresolvedReferences
    def test_nn_normal_conditions(self):
        index = DummySI()
        # Need to force a non-zero index size for knn to be performed.
        index.count = mock.MagicMock()
        index.count.return_value = 1

        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        # Basically this shouldn't crash
        index.nn(q)

    def test_query_empty_index(self):
        index = DummySI()
        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        self.assertRaises(ValueError, index.nn, q)
