import pytest
import unittest.mock as mock

from smqtk.algorithms import RelevancyIndex
from smqtk.iqr import IqrSession
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement


class TestIqrSession (object):
    """
    Unit tests pertaining to the IqrSession class.
    """

    def test_adjudicate_new_pos_neg(self):
        """
        Test that providing iterables to ``new_positives`` and
        ``new_negatives`` parameters result in additions to the positive and
        negative sets respectively.
        """
        iqrs = IqrSession()

        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        iqrs.adjudicate(new_positives=[p0])
        assert iqrs.positive_descriptors == {p0}
        assert iqrs.negative_descriptors == set()

        n1 = DescriptorMemoryElement('', 1).set_vector([1])
        iqrs.adjudicate(new_negatives=[n1])
        assert iqrs.positive_descriptors == {p0}
        assert iqrs.negative_descriptors == {n1}

        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        p3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])
        iqrs.adjudicate(new_positives=[p2, p3], new_negatives=[n4])
        assert iqrs.positive_descriptors == {p0, p2, p3}
        assert iqrs.negative_descriptors == {n1, n4}

    def test_adjudicate_add_duplicates(self):
        """
        Test that adding duplicate descriptors as positive or negative
        adjudications has no effect as the behavior of sets should be observed.
        """
        iqrs = IqrSession()

        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n1 = DescriptorMemoryElement('', 1).set_vector([1])
        p3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])

        # Partially add the above descriptors
        iqrs.adjudicate(new_positives=[p0], new_negatives=[n1])
        assert iqrs.positive_descriptors == {p0}
        assert iqrs.negative_descriptors == {n1}

        # Add all descriptors, observing that that already added descriptors
        # are ignored.
        iqrs.adjudicate(new_positives=[p0, p2, p3], new_negatives=[n1, n4])
        assert iqrs.positive_descriptors == {p0, p2, p3}
        assert iqrs.negative_descriptors == {n1, n4}

        # Duplicate previous call so no new descriptors are added. No change or
        # issue should be observed.
        iqrs.adjudicate(new_positives=[p0, p2, p3], new_negatives=[n1, n4])
        assert iqrs.positive_descriptors == {p0, p2, p3}
        assert iqrs.negative_descriptors == {n1, n4}

    def test_adjudication_switch(self):
        """
        Test providing positives and negatives on top of an existing state such
        that the descriptor adjudications are reversed. (what was once positive
        is now negative, etc.)
        """
        iqrs = IqrSession()

        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p1 = DescriptorMemoryElement('', 1).set_vector([1])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])

        # Set initial state
        iqrs.positive_descriptors = {p0, p1, p2}
        iqrs.negative_descriptors = {n3, n4}

        # Adjudicate, partially swapping adjudications individually
        iqrs.adjudicate(new_positives=[n3])
        assert iqrs.positive_descriptors == {p0, p1, p2, n3}
        assert iqrs.negative_descriptors == {n4}

        iqrs.adjudicate(new_negatives=[p1])
        assert iqrs.positive_descriptors == {p0, p2, n3}
        assert iqrs.negative_descriptors == {n4, p1}

        # Adjudicate swapping remaining at the same time
        iqrs.adjudicate(new_positives=[n4], new_negatives=[p0, p2])
        assert iqrs.positive_descriptors == {n3, n4}
        assert iqrs.negative_descriptors == {p0, p1, p2}

    def test_adjudicate_remove_pos_neg(self):
        """
        Test that we can remove positive and negative adjudications using
        "un_*" parameters.
        """
        iqrs = IqrSession()

        # Set initial state
        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p1 = DescriptorMemoryElement('', 1).set_vector([1])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])

        # Set initial state
        iqrs.positive_descriptors = {p0, p1, p2}
        iqrs.negative_descriptors = {n3, n4}

        # "Un-Adjudicate" descriptors individually
        iqrs.adjudicate(un_positives=[p1])
        assert iqrs.positive_descriptors == {p0, p2}
        assert iqrs.negative_descriptors == {n3, n4}
        iqrs.adjudicate(un_negatives=[n3])
        assert iqrs.positive_descriptors == {p0, p2}
        assert iqrs.negative_descriptors == {n4}

        # "Un-Adjudicate" collectively
        iqrs.adjudicate(un_positives=[p0, p2], un_negatives=[n4])
        assert iqrs.positive_descriptors == set()
        assert iqrs.negative_descriptors == set()

    def test_adjudicate_combined_remove_unadj(self):
        """
        Test combining adjudication switching with un-adjudication.
        """
        iqrs = IqrSession()

        # Set initial state
        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p1 = DescriptorMemoryElement('', 1).set_vector([1])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])

        # Set initial state
        iqrs.positive_descriptors = {p0, p1, p2}
        iqrs.negative_descriptors = {n3, n4}

        # Add p5, switch p1 to negative, unadj p2
        p5 = DescriptorMemoryElement('', 5).set_vector([5])
        iqrs.adjudicate(new_positives=[p5], new_negatives=[p1],
                        un_positives=[p2])
        assert iqrs.positive_descriptors == {p0, p5}
        assert iqrs.negative_descriptors == {n3, n4, p1}

        # Add n6, switch n4 to positive, unadj n3
        n6 = DescriptorMemoryElement('', 6).set_vector([6])
        iqrs.adjudicate(new_positives=[n4], new_negatives=[n6],
                        un_negatives=[n3])
        assert iqrs.positive_descriptors == {p0, p5, n4}
        assert iqrs.negative_descriptors == {p1, n6}

    def test_adjudicate_both_labels(self):
        """
        Test that providing a descriptor element as both a positive AND
        negative adjudication causes no state change..
        """
        iqrs = IqrSession()

        # Set initial state
        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p1 = DescriptorMemoryElement('', 1).set_vector([1])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])

        # Set initial state
        iqrs.positive_descriptors = {p0, p1, p2}
        iqrs.negative_descriptors = {n3, n4}

        # Attempt adjudicating a new element as both postive AND negative
        e = DescriptorMemoryElement('', 5).set_vector([5])
        iqrs.adjudicate(new_positives=[e], new_negatives=[e])
        assert iqrs.positive_descriptors == {p0, p1, p2}
        assert iqrs.negative_descriptors == {n3, n4}

    def test_adjudicate_unadj_noeffect(self):
        """
        Test that an empty call, or un-adjudicating a descriptor that is not
        currently marked as a positive or negative, causes no state change.
        """
        iqrs = IqrSession()

        # Set initial state
        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p1 = DescriptorMemoryElement('', 1).set_vector([1])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])

        # Set initial state
        iqrs.positive_descriptors = {p0, p1, p2}
        iqrs.negative_descriptors = {n3, n4}

        # Empty adjudication
        iqrs.adjudicate()
        assert iqrs.positive_descriptors == {p0, p1, p2}
        assert iqrs.negative_descriptors == {n3, n4}

        # Attempt un-adjudication of a non-adjudicated element.
        e = DescriptorMemoryElement('', 5).set_vector([5])
        iqrs.adjudicate(un_positives=[e], un_negatives=[e])
        assert iqrs.positive_descriptors == {p0, p1, p2}
        assert iqrs.negative_descriptors == {n3, n4}

    def test_adjudicate_cache_resetting_positive(self):
        """
        Test results view cache resetting functionality on adjudicating certain
        ways.
        """
        e = DescriptorMemoryElement('', 0).set_vector([0])

        iqrs = IqrSession()
        iqrs._ordered_pos = True
        iqrs._ordered_neg = True
        iqrs._ordered_non_adj = True

        # Check that adding a positive adjudication resets the positive and
        # non-adjudicated result caches.
        iqrs.adjudicate(new_positives=[e])
        assert iqrs._ordered_pos is None  # reset
        assert iqrs._ordered_neg is True  # NOT reset
        assert iqrs._ordered_non_adj is None  # reset

    def test_adjudicate_cache_resetting_negative(self):
        """
        Test results view cache resetting functionality on adjudicating certain
        ways.
        """
        e = DescriptorMemoryElement('', 0).set_vector([0])

        iqrs = IqrSession()
        iqrs._ordered_pos = True
        iqrs._ordered_neg = True
        iqrs._ordered_non_adj = True

        # Check that adding a positive adjudication resets the positive and
        # non-adjudicated result caches.
        iqrs.adjudicate(new_negatives=[e])
        assert iqrs._ordered_pos is True  # NOT reset
        assert iqrs._ordered_neg is None  # reset
        assert iqrs._ordered_non_adj is None  # reset

    def test_adjudication_cache_not_reset(self):
        """
        Test that pos/neg/non-adj result caches are NOT reset when no state
        change occurs under different circumstances
        """
        # setup initial IQR session state.
        p0 = DescriptorMemoryElement('', 0).set_vector([0])
        p1 = DescriptorMemoryElement('', 1).set_vector([1])
        p2 = DescriptorMemoryElement('', 2).set_vector([2])
        n3 = DescriptorMemoryElement('', 3).set_vector([3])
        n4 = DescriptorMemoryElement('', 4).set_vector([4])
        iqrs = IqrSession()
        iqrs.positive_descriptors = {p0, p1, p2}
        iqrs.negative_descriptors = {n3, n4}
        iqrs._ordered_pos = iqrs._ordered_neg = iqrs._ordered_non_adj = True

        # Empty adjudication
        iqrs.adjudicate()
        assert iqrs._ordered_pos is True
        assert iqrs._ordered_neg is True
        assert iqrs._ordered_non_adj is True

        # Repeat positive/negative adjudication
        iqrs.adjudicate(new_positives=[p0])
        assert iqrs._ordered_pos is True
        assert iqrs._ordered_neg is True
        assert iqrs._ordered_non_adj is True
        iqrs.adjudicate(new_negatives=[n3])
        assert iqrs._ordered_pos is True
        assert iqrs._ordered_neg is True
        assert iqrs._ordered_non_adj is True
        iqrs.adjudicate(new_positives=[p1], new_negatives=[n4])
        assert iqrs._ordered_pos is True
        assert iqrs._ordered_neg is True
        assert iqrs._ordered_non_adj is True

        # No-op un-adjudication
        e = DescriptorMemoryElement('', 5).set_vector([5])
        iqrs.adjudicate(un_positives=[e], un_negatives=[e])
        assert iqrs._ordered_pos is True
        assert iqrs._ordered_neg is True
        assert iqrs._ordered_non_adj is True

    def test_refine_no_rel_index(self):
        """
        Test that refinement cannot occur if there is no relevancy index
        instance yet.
        """
        iqrs = IqrSession()
        with pytest.raises(RuntimeError, match="No relevancy index yet"):
            iqrs.refine()

    def test_refine_no_pos(self):
        """
        Test that refinement cannot occur if there are no positive descriptor
        external/adjudicated elements.
        """
        iqrs = IqrSession()
        # Mock relevancy index in order to check how its called and mock return
        # value.
        iqrs.rel_index = mock.MagicMock(spec=RelevancyIndex)
        # Mock length to be non-zero to simulate it having contents
        iqrs.rel_index.__len__.return_value = 1

        with pytest.raises(RuntimeError, match='Did not find at least one '
                                               'positive adjudication'):
            iqrs.refine()

    def test_refine_no_prev_results(self):
        """
        Test that the results of RelevancyIndex ranking are directly reflected
        in a new results dictionary of probability values, even for elements
        that were also used in adjudication.

        This test is useful because a previous state of the IQR Session
        structure would force return probabilities for some descriptor elements
        to certain values if they were also present in the positive or negative
        adjudicate (internal or external) sets.
        """
        # IqrSession instance. No config for rel_index because we will mock
        # that.
        iqrs = IqrSession()
        # Mock relevancy index in order to check how its called and mock return
        # value.
        iqrs.rel_index = mock.MagicMock(spec=RelevancyIndex)
        # Mock length to be non-zero to simulate it having contents
        iqrs.rel_index.__len__.return_value = 1

        test_in_pos_elem = DescriptorMemoryElement('t', 0).set_vector([0])
        test_in_neg_elem = DescriptorMemoryElement('t', 1).set_vector([1])
        test_ex_pos_elem = DescriptorMemoryElement('t', 2).set_vector([2])
        test_ex_neg_elem = DescriptorMemoryElement('t', 3).set_vector([3])
        test_other_elem = DescriptorMemoryElement('t', 4).set_vector([4])

        # Mock return dictionary, probabilities don't matter much other than
        # they are not 1.0 or 0.0.
        iqrs.rel_index.rank.return_value = \
            {e: 0.5 for e in [test_in_pos_elem, test_in_neg_elem,
                              test_other_elem]}

        # Asserting expected pre-condition where there are no results yet.
        assert iqrs.results is None

        # Prepare IQR state for refinement
        # - set dummy internal/external positive negatives.
        iqrs.external_descriptors(
            positive=[test_ex_pos_elem], negative=[test_ex_neg_elem]
        )
        iqrs.adjudicate(
            new_positives=[test_in_pos_elem], new_negatives=[test_in_neg_elem]
        )

        # Test calling refine method
        iqrs.refine()

        # We test that:
        # - ``rel_index.rank`` called with the combination of
        #   external/adjudicated descriptor elements.
        # - ``results`` attribute now has a dict value
        # - value of ``results`` attribute is what we expect.
        iqrs.rel_index.rank.assert_called_once_with(
            {test_in_pos_elem, test_ex_pos_elem},
            {test_in_neg_elem, test_ex_neg_elem},
        )
        assert iqrs.results is not None
        assert len(iqrs.results) == 3
        assert test_other_elem in iqrs.results
        assert test_in_pos_elem in iqrs.results
        assert test_in_neg_elem in iqrs.results

        assert iqrs.results[test_other_elem] == 0.5
        assert iqrs.results[test_in_pos_elem] == 0.5
        assert iqrs.results[test_in_neg_elem] == 0.5

    def test_refine_with_prev_results(self):
        """
        Test that the results of RelevancyIndex ranking are directly reflected
        in an existing results dictionary of probability values.
        """
        # IqrSession instance. No config for rel_index because we will mock
        # that.
        iqrs = IqrSession()
        # Mock relevancy index in order to check how its called and mock return
        # value.
        iqrs.rel_index = mock.MagicMock(spec=RelevancyIndex)
        # Mock length to be non-zero to simulate it having contents
        iqrs.rel_index.__len__.return_value = 1

        test_in_pos_elem = DescriptorMemoryElement('t', 0).set_vector([0])
        test_in_neg_elem = DescriptorMemoryElement('t', 1).set_vector([1])
        test_ex_pos_elem = DescriptorMemoryElement('t', 2).set_vector([2])
        test_ex_neg_elem = DescriptorMemoryElement('t', 3).set_vector([3])
        test_other_elem = DescriptorMemoryElement('t', 4).set_vector([4])

        # Mock return dictionary, probabilities don't matter much other than
        # they are not 1.0 or 0.0.
        iqrs.rel_index.rank.return_value = \
            {e: 0.5 for e in [test_in_pos_elem, test_in_neg_elem,
                              test_other_elem]}

        # Create a "previous state" of the results dictionary containing
        # results from our "working set" of descriptor elements.
        iqrs.results = {
            test_in_pos_elem: 0.2,
            test_in_neg_elem: 0.2,
            test_other_elem: 0.2,
            # ``refine`` replaces the previous dict, so disjoint keys are
            # NOT retained.
            'something else': 0.3,
        }

        # Prepare IQR state for refinement
        # - set dummy internal/external positive negatives.
        iqrs.external_descriptors(
            positive=[test_ex_pos_elem], negative=[test_ex_neg_elem]
        )
        iqrs.adjudicate(
            new_positives=[test_in_pos_elem], new_negatives=[test_in_neg_elem]
        )

        # Test calling refine method
        iqrs.refine()

        # We test that:
        # - ``rel_index.rank`` called with the combination of
        #   external/adjudicated descriptor elements.
        # - ``results`` attribute now has an dict value
        # - value of ``results`` attribute is what we expect.
        iqrs.rel_index.rank.assert_called_once_with(
            {test_in_pos_elem, test_ex_pos_elem},
            {test_in_neg_elem, test_ex_neg_elem},
        )
        assert iqrs.results is not None
        assert len(iqrs.results) == 3
        assert test_other_elem in iqrs.results
        assert test_in_pos_elem in iqrs.results
        assert test_in_neg_elem in iqrs.results
        assert 'something else' not in iqrs.results

        assert iqrs.results[test_other_elem] == 0.5
        assert iqrs.results[test_in_pos_elem] == 0.5
        assert iqrs.results[test_in_neg_elem] == 0.5

    def test_ordered_results_no_results_no_cache(self):
        """
        Test that an empty list is returned when ``ordered_results`` is called
        before any refinement has occurred.
        """
        iqrs = IqrSession()
        assert iqrs.ordered_results() == []

    def test_ordered_results_has_cache(self):
        """
        Test that a shallow copy of the cached list is returned when there is
        a cache.
        """
        iqrs = IqrSession()
        # Simulate there being a cache
        iqrs._ordered_pos = ['simulated', 'cache']
        actual = iqrs.get_positive_adjudication_relevancy()
        assert actual == iqrs._ordered_pos
        assert id(actual) != id(iqrs._ordered_pos)

    def test_ordered_results_has_results_no_cache(self):
        """
        Test that an appropriate list is returned by ``ordered_results`` after
        a refinement has occurred.
        """
        iqrs = IqrSession()

        # Mocking results map existing for return.
        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        d3 = DescriptorMemoryElement('', 3).set_vector([3])
        iqrs.results = {
            d0: 0.0,
            d1: 0.8,
            d2: 0.2,
            d3: 0.4,
        }

        # Cache should be empty before call to ``ordered_results``
        assert iqrs._ordered_results is None

        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual1 = iqrs.ordered_results()
            m_sorted.assert_called_once()

        expected = [(d1, 0.8), (d3, 0.4), (d2, 0.2), (d0, 0.0)]
        assert actual1 == expected

        # Calling the method a second time should not result in a ``sorted``
        # operation due to caching.
        with mock.patch('smqtk.iqr.iqr_session.sorted') as m_sorted:
            actual2 = iqrs.ordered_results()
            m_sorted.assert_not_called()

        assert actual2 == expected
        # Both returns should be shallow copies, thus not the same list
        # instances.
        assert id(actual1) != id(actual2)

    def test_ordered_results_has_results_post_reset(self):
        """
        Test that an empty list is returned after a reset where there was a
        cached value before the reset.
        """
        iqrs = IqrSession()

        # Mocking results map existing for return.
        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        d3 = DescriptorMemoryElement('', 3).set_vector([3])
        iqrs.results = {
            d0: 0.0,
            d1: 0.8,
            d2: 0.2,
            d3: 0.4,
        }

        # Initial call to ``ordered_results`` should have a non-None return.
        assert iqrs.ordered_results() is not None

        iqrs.reset()

        # Post-reset, there should be no results nor cache.
        actual = iqrs.ordered_results()
        assert actual == []

    def test_get_positive_adjudication_relevancy_has_cache(self):
        """
        Test that a shallow copy of the cached list is returned if there is a
        cache.
        """
        iqrs = IqrSession()

        iqrs._ordered_pos = ['simulation', 'cache']
        actual = iqrs.get_positive_adjudication_relevancy()
        assert actual == ['simulation', 'cache']
        assert id(actual) != id(iqrs._ordered_pos)

    def test_get_positive_adjudication_relevancy_no_cache_no_results(self):
        """
        Test that ``get_positive_adjudication_relevancy`` returns None when in a
        pre-refine state when there are no positive adjudications.
        """
        iqrs = IqrSession()
        assert iqrs.get_positive_adjudication_relevancy() == []

    def test_get_positive_adjudication_relevancy_no_cache_has_results(self):
        """
        Test that we can get positive adjudication relevancy scores correctly
        from a not-cached state.
        """
        iqrs = IqrSession()

        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        d3 = DescriptorMemoryElement('', 3).set_vector([3])

        # Simulate a populated contributing adjudication state (there must be
        # some positives for a simulated post-refine state to be valid).
        iqrs.rank_contrib_pos = {d1, d3}
        iqrs.rank_contrib_neg = {d0}

        # Simulate post-refine results map.
        iqrs.results = {
            d0: 0.1,
            d1: 0.8,
            d2: 0.2,
            d3: 0.4,
        }

        # Cache is initially empty
        assert iqrs._ordered_pos is None

        # Test that the appropriate sorting actually occurs.
        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual1 = iqrs.get_positive_adjudication_relevancy()
            m_sorted.assert_called_once()

        expected = [(d1, 0.8), (d3, 0.4)]
        assert actual1 == expected

        # Calling the method a second time should not result in a ``sorted``
        # operation due to caching.
        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual2 = iqrs.get_positive_adjudication_relevancy()
            m_sorted.assert_not_called()

        assert actual2 == expected
        # Both returns should be shallow copies, thus not the same list
        # instances.
        assert id(actual1) != id(actual2)

    def test_get_negative_adjudication_relevancy_has_cache(self):
        """
        Test that a shallow copy of the cached list is returned if there is a
        cache.
        """
        iqrs = IqrSession()

        iqrs._ordered_neg = ['simulation', 'cache']
        actual = iqrs.get_negative_adjudication_relevancy()
        assert actual == ['simulation', 'cache']
        assert id(actual) != id(iqrs._ordered_neg)

    def test_get_negative_adjudication_relevancy_no_cache_no_results(self):
        """
        Test that ``get_negative_adjudication_relevancy`` returns None when in a
        pre-refine state when there are no negative adjudications.
        """
        iqrs = IqrSession()
        assert iqrs.get_negative_adjudication_relevancy() == []

    def test_get_negative_adjudication_relevancy_no_cache_has_results(self):
        """
        Test that we can get negative adjudication relevancy scores correctly
        from a not-cached state.
        """
        iqrs = IqrSession()

        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        d3 = DescriptorMemoryElement('', 3).set_vector([3])

        # Simulate a populated contributing adjudication state (there must be
        # some positives for a simulated post-refine state to be valid).
        iqrs.rank_contrib_pos = {d1}
        iqrs.rank_contrib_neg = {d0, d2}

        # Simulate post-refine results map.
        iqrs.results = {
            d0: 0.1,
            d1: 0.8,
            d2: 0.2,
            d3: 0.4,
        }

        # Cache is initially empty
        assert iqrs._ordered_neg is None

        # Test that the appropriate sorting actually occurs.
        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual1 = iqrs.get_negative_adjudication_relevancy()
            m_sorted.assert_called_once()

        expected = [(d2, 0.2), (d0, 0.1)]
        assert actual1 == expected

        # Calling the method a second time should not result in a ``sorted``
        # operation due to caching.
        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual2 = iqrs.get_negative_adjudication_relevancy()
            m_sorted.assert_not_called()

        assert actual2 == expected
        # Both returns should be shallow copies, thus not the same list
        # instances.
        assert id(actual1) != id(actual2)

    def test_get_unadjudicated_relevancy_has_cache(self):
        """
        Test that a shallow copy of the cached list is returned if there is a
        cache.
        """
        iqrs = IqrSession()

        iqrs._ordered_non_adj = ['simulation', 'cache']
        actual = iqrs.get_unadjudicated_relevancy()
        assert actual == ['simulation', 'cache']
        assert id(actual) != id(iqrs._ordered_non_adj)

    def test_get_unadjudicated_relevancy_no_cache_no_results(self):
        """
        Test that ``get_unadjudicated_relevancy`` returns None when in a
        pre-refine state when there is results state.
        """
        iqrs = IqrSession()
        assert iqrs.get_unadjudicated_relevancy() == []

    def test_get_unadjudicated_relevancy_no_cache_has_results(self):
        """
        Test that we get the non-adjudicated DescriptorElements and their
        scores correctly from a non-cached state with known results.
        """
        iqrs = IqrSession()

        d0 = DescriptorMemoryElement('', 0).set_vector([0])
        d1 = DescriptorMemoryElement('', 1).set_vector([1])
        d2 = DescriptorMemoryElement('', 2).set_vector([2])
        d3 = DescriptorMemoryElement('', 3).set_vector([3])

        # Simulate a populated contributing adjudication state (there must be
        # some positives for a simulated post-refine state to be valid).
        iqrs.rank_contrib_pos = {d1}
        iqrs.rank_contrib_neg = {d0}

        # Simulate post-refine results map.
        iqrs.results = {
            d0: 0.1,
            d1: 0.8,
            d2: 0.2,
            d3: 0.4,
        }

        # Cache should be initially empty
        assert iqrs._ordered_non_adj is None

        # Test that the appropriate sorting actually occurs.
        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual1 = iqrs.get_unadjudicated_relevancy()
            m_sorted.assert_called_once()

        expected = [(d3, 0.4), (d2, 0.2)]
        assert actual1 == expected

        # Calling the method a second time should not result in a ``sorted``
        # operation due to caching.
        with mock.patch('smqtk.iqr.iqr_session.sorted',
                        side_effect=sorted) as m_sorted:
            actual2 = iqrs.get_unadjudicated_relevancy()
            m_sorted.assert_not_called()

        assert actual2 == expected
        # Both returns should be shallow copies, thus not the same list
        # instances.
        assert id(actual1) != id(actual2)

    def test_reset_result_cache_invalidation(self):
        """
        Test that calling the reset method resets the result view caches to
        None.
        """
        # Setup initial IQR session state
        iqrs = IqrSession()
        iqrs._ordered_pos = iqrs._ordered_neg = iqrs._ordered_non_adj = True

        iqrs.reset()
        assert iqrs._ordered_pos is None
        assert iqrs._ordered_neg is None
        assert iqrs._ordered_non_adj is None


class TestIqrSessionBehavior (object):
    """
    Test certain IqrSession state transitions
    """
    # TODO - More complicated state transitions.
