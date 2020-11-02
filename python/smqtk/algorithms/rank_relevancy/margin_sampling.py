from typing import Hashable, Sequence, Tuple

from numpy import ndarray

from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from smqtk.utils.dict import merge_dict
from ._interface import RankRelevancy, RankRelevancyWithFeedback


class RankRelevancyWithMarginSampledFeedback(RankRelevancyWithFeedback):
    """
    Wrap an instance of :class:`RankRelevancy` to provide feedback via
    margin sampling

    :param rank_relevancy: :class:`RankRelevancy` to use for computing
        relevancy scores
    :param n: Maximum number of items to return for feedback
    :param center: Value for which pool items whose relevancy score is
        closest to it will be returned for feedback (default: 0.5)

    :raises ValueError: n is negative

    """

    def __init__(self, rank_relevancy: RankRelevancy,
                 n: int, center: float = 0.5):
        self._rank_relevancy = rank_relevancy
        if n < 0:
            raise ValueError(f"n must be nonnegative but got {n}")
        self._n = n
        self._center = center

    def _rank_with_feedback(
            self,
            pos: Sequence[ndarray],
            neg: Sequence[ndarray],
            pool: Sequence[ndarray],
            pool_uids: Sequence[Hashable],
    ) -> Tuple[Sequence[float], Sequence[Hashable]]:
        scores = self._rank_relevancy.rank(pos, neg, pool)
        c = self._center
        ranked = sorted(zip(scores, pool_uids), key=lambda su: abs(su[0] - c))
        return scores, [r[1] for r in ranked[:self._n]]

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        config_dict = dict(config_dict, rank_relevancy=from_config_dict(
            config_dict['rank_relevancy'], RankRelevancy.get_impls(),
        ))
        return super().from_config(config_dict, merge_default=merge_default)

    @classmethod
    def get_default_config(cls):
        c = super().get_default_config()
        rr_default = make_default_config(RankRelevancy.get_impls())
        return dict(c, rank_relevancy=rr_default)

    def get_config(self):
        return merge_dict(self.get_default_config(), dict(
            rank_relevancy=to_config_dict(self._rank_relevancy),
            n=self._n,
            center=self._center,
        ))
