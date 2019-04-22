import pytest
from six.moves import mock

from smqtk.algorithms import RelevancyIndex
from smqtk.iqr import IqrSession
from smqtk.iqr.iqr_session import IqrResultsDict
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement


class TestIqrSession (object):
    """
    Unit tests pertaining to the IqrSession class.
    """

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
        # If results was -> None: a new IqrResultsDict should be created.
        #                   something: Should be

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
        # - ``results`` attribute now has an IqrResultsDict value
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
        # If results was -> None: a new IqrResultsDict should be created.
        #                   something: Should be

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
        iqrs.results = IqrResultsDict({
            test_in_pos_elem: 0.2,
            test_in_neg_elem: 0.2,
            test_other_elem: 0.2,
            # ``refine`` updates the existing dict, so disjoint keys are
            # retained.
            'something else': 0.3,
        })

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
        # - ``results`` attribute now has an IqrResultsDict value
        # - value of ``results`` attribute is what we expect.
        iqrs.rel_index.rank.assert_called_once_with(
            {test_in_pos_elem, test_ex_pos_elem},
            {test_in_neg_elem, test_ex_neg_elem},
        )
        assert iqrs.results is not None
        assert len(iqrs.results) == 4
        assert test_other_elem in iqrs.results
        assert test_in_pos_elem in iqrs.results
        assert test_in_neg_elem in iqrs.results
        assert 'something else' in iqrs.results

        assert iqrs.results[test_other_elem] == 0.5
        assert iqrs.results[test_in_pos_elem] == 0.5
        assert iqrs.results[test_in_neg_elem] == 0.5
        assert iqrs.results['something else'] == 0.3
