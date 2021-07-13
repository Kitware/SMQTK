import numpy

from smqtk.algorithms import RankRelevancy
from smqtk.algorithms.rank_relevancy.margin_sampling import (
    RankRelevancyWithMarginSampledFeedback,
)


def test_is_usable():
    assert RankRelevancyWithMarginSampledFeedback.is_usable()


class DummyRankRelevancy(RankRelevancy):
    def get_config(self):
        return {}

    def rank(self, pos, neg, pool):
        return [v[0] for v in pool]


def make_margin_ranker(n, center=None):
    return RankRelevancyWithMarginSampledFeedback(
        DummyRankRelevancy(), n, *(() if center is None else [center]),
    )


def test_parameter_n():
    """
    Check that the "n" parameter has the expected effect on feedback
    request count
    """
    n = 10
    mr = make_margin_ranker(n)
    for i in range(3, 31, 3):
        pool = numpy.linspace(0, 1, i)[:, numpy.newaxis]
        pool_uids = [object() for _ in range(i)]
        scores, requests = mr.rank_with_feedback([], [], pool, pool_uids)
        assert len(requests) == min(n, i)


def test_pass_through():
    """
    Check that scores from the wrapped RankRelevancy are passed
    through
    """
    mr = make_margin_ranker(3)
    pool = [[.3], [.1], [.45], [.29], [.03]]
    expected = [.3, .1, .45, .29, .03]
    uids = [object() for _ in pool]
    scores, requests = mr.rank_with_feedback([], [], pool, uids)
    assert list(scores) == expected


def test_parameter_center():
    """
    Check that the "center" parameter has the expected effect on the
    choice of feedback requests
    """
    pool = [[i ** 2 / 100] for i in range(11)]
    uids = 'abcdefghijk'
    centers = [.045, .2]
    expected = ['cb', 'ef']  # [(.04, .01), (.16, .25)]
    for c, e in zip(centers, expected):
        mr = make_margin_ranker(2, c)
        scores, requests = mr.rank_with_feedback([], [], pool, uids)
        assert list(requests) == list(e)
