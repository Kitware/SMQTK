import abc
from typing import Hashable, Sequence, Tuple

from numpy import ndarray

from smqtk.algorithms import SmqtkAlgorithm


class RankRelevancy (SmqtkAlgorithm):
    """
    Algorithm that can rank a given pool of descriptors based on positively
    and negatively adjudicated descriptors.
    """

    @abc.abstractmethod
    def rank(
            self,
            pos: Sequence[ndarray],
            neg: Sequence[ndarray],
            pool: Sequence[ndarray],
    ) -> Sequence[float]:
        """
        Assign a relevancy score to each input descriptor in `pool` based on
        the positively and negatively adjudicated descriptors in `pos` and
        `neg` respectively.

        :param pos:
            Sequence of positively adjudicated descriptor vectors.
        :param neg:
            Sequence of negatively adjudicated descriptor vectors.
        :param pool:
            A sequence of descriptor vectors that we want to rank by topical
            relevancy relative to the given positive and negative examples.

        :return: An ordered sequence of float values denoting the relevancy of
            `pool` elements
        """


class RankRelevancyWithFeedback (SmqtkAlgorithm):
    """
    Similar to the :class:`RankRelevancy` algorithm but with the added feature
    of also returning a sequence of elements from which feedback would be "most
    useful".

    What "most useful" means may be flexible but generally refers to the
    goal of reducing the amount of adjudications required in order to
    separate true-positive examples from true-negative examples in provided
    pools via the assigned relevancy scores. E.g. other elements may be
    adjudicated in some quantity to achieve some level of relevant sample
    separation, but if the feedback requests are instead adjudicated, less
    elements may need to be adjudicated to achieve and equivalent level of
    separation.

    Feedback requests ought to be returned in a form that is meaningful for the
    user to be able to properly convey the proper information to the
    adjudicating agent to actually perform adjudications. Additionally, we want
    to be able to request feedback from elements that may not be present in the
    given pool of descriptors.

    Towards that end, this algorithm should be given a sequence of UIDs for the
    given pool of descriptors. This allows the implementation to potentially
    coordinate with an outside source of descriptor references such that the
    returned feedback requests may be interpreted uniformly.
    """

    @abc.abstractmethod
    def _rank_with_feedback(
            self,
            pos: Sequence[ndarray],
            neg: Sequence[ndarray],
            pool: Sequence[ndarray],
            pool_uids: Sequence[Hashable],
    ) -> Tuple[Sequence[float], Sequence[Hashable]]:
        """
        Implement :meth:`rank_with_feedback`.  `pool` and `pool_uids` have
        already been checked to be of equal length.

        .. seealso:: :meth:`rank_with_feedback`'s doc-string for the meanings
           of the parameters and their return values
        """

    def rank_with_feedback(
            self,
            pos: Sequence[ndarray],
            neg: Sequence[ndarray],
            pool: Sequence[ndarray],
            pool_uids: Sequence[Hashable],
    ) -> Tuple[Sequence[float], Sequence[Hashable]]:
        """
        Assign a relevancy score to each input descriptor in `pool` based on
        the positively and negatively adjudicated descriptors in `pos` and
        `neg` respectively, additionally returning a sequence of UIDs of those
        descriptors for which adjudication feedback would be "most useful".

        :param pos:
            Sequence of positively adjudicated descriptor vectors.
        :param neg:
            Sequence of negatively adjudicated descriptor vectors.
        :param pool:
            A sequence of descriptor vectors that we want to rank by topical
            relevancy relative to the given positive and negative examples.
        :param pool_uids:
            A sequence of hashable UID values, parallel in association with
            descriptors in `pool`.

        :return: Ordered sequence of float values denoting relevancy of `pool`
            elements, as well as a sequence of ``Hashable`` values referencing
            in-pool or out-of-pool descriptors we recommend for adjudication
            feedback.  In the latter sequence, descriptors are ordered
            by usefulness, most to least.

        :raises ValueError: `pool` and `pool_uids` are of different length

        .. seealso:: :py:class:`RankRelevancyWithFeedback` class doc-string for
            discussion on "most useful" meaning.
        """
        if len(pool) != len(pool_uids):
            raise ValueError('pool and pool_uids must be equally long but '
                             f'have length {len(pool)} and {len(pool_uids)}, '
                             'respectively')
        return self._rank_with_feedback(pos, neg, pool, pool_uids)
