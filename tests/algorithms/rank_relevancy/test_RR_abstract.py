from unittest import mock

import numpy
import pytest

from smqtk.algorithms import RankRelevancyWithFeedback


class DummyRRWF(RankRelevancyWithFeedback):
    def __init__(self):
        self._rwf_mock = mock.Mock()

    # Implement RankRelevancyWithFeedback._rank_with_feedback using a
    # per-instance Mock.  self._rank_with_feedback is thus a Mock
    # instead of a bound method.
    @property
    def _rank_with_feedback(self):
        return self._rwf_mock

    def get_config(self):
        raise NotImplementedError


def test_rrwf_length_check():
    """
    Check that :meth:`RankRelevancyWithFeedback.rank_with_feedback`
    raises the documented :class:`ValueError` when pool and UID list
    length don't match
    """
    rrwf = DummyRRWF()
    with pytest.raises(ValueError):
        v = numpy.ones(16)
        rrwf.rank_with_feedback(
            [v] * 4, [v] * 4, [v] * 10,
            # Not 10 UIDs
            ['a', 'b', 'c', 'd'],
        )
    rrwf._rank_with_feedback.assert_not_called()


def test_rrwf_calls_impl_method():
    """
    Check that :meth:`RankRelevancyWithFeedback.rank_with_feedback`
    delegates to :meth:`RankRelevancyWithFeedback._rank_with_feedback`
    """
    rrwf = DummyRRWF()
    v = numpy.ones(16)
    args = (
        [v] * 4, [v] * 4, [v] * 10,
        # 10 UIDs
        list('abcdefghij'),
    )
    result = rrwf.rank_with_feedback(*args)
    rrwf._rank_with_feedback.assert_called_once_with(*args)
    assert result is rrwf._rank_with_feedback.return_value
