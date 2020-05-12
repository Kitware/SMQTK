import functools
import unittest

import unittest.mock as mock
import numpy as np
import pytest

from smqtk.algorithms.relevancy_index.classifier_wrapper import \
    SupervisedClassifierRelevancyIndex, NoIndexError
from smqtk.algorithms import SupervisedClassifier
from smqtk.representation import DescriptorElement
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


class TestSupervisedClassifierRelevancyIndex (unittest.TestCase):
    """
    Unit tests for the SupervisedClassifierRelevancyIndex implementation.
    """

    def test_is_usable(self):
        """ test that this impl is always available. """
        assert SupervisedClassifierRelevancyIndex.is_usable() is True

    def test_rank_before_build(self):
        """ Test that the appropriate exception occurs if attempting to rank
        before building the index."""
        m_classifier_inst = mock.MagicMock(spec=SupervisedClassifier)

        m_pos_elems = [mock.MagicMock(spec=DescriptorElement)]
        m_neg_elems = [mock.MagicMock(spec=DescriptorElement)]

        ri = SupervisedClassifierRelevancyIndex(m_classifier_inst)

        with pytest.raises(NoIndexError):
            ri.rank(pos=m_pos_elems, neg=m_neg_elems)

    def test_empty_count(self):
        """ Test that count is 0 before a build. """
        m_classifier_inst = mock.MagicMock(spec=SupervisedClassifier)
        ri = SupervisedClassifierRelevancyIndex(m_classifier_inst)
        assert ri.count() == 0

    def test_build_index(self):
        """ Test "building" a generic index, i.e. descriptor corpus. """
        m_classifier_inst = mock.MagicMock(spec=SupervisedClassifier)
        ri = SupervisedClassifierRelevancyIndex(m_classifier_inst)

        build_elems = [DescriptorMemoryElement('t', i).set_vector([i])
                       for i in range(10)]
        ri.build_index(build_elems)

        assert ri.count() == 10
        assert ri._descr_elem_list == build_elems
        assert np.allclose(ri._descr_matrix, [[i] for i in range(10)])

    def test_build_index_no_elements(self):
        """ Test that the expected error occurs when attempting a build with no
        elements."""
        m_classifier_inst = mock.MagicMock(spec=SupervisedClassifier)
        ri = SupervisedClassifierRelevancyIndex(m_classifier_inst)
        with pytest.raises(ValueError, match="No descriptor elements passed"):
            ri.build_index(iter([]))

    def test_rank(self):
        """
        Test wrapper ``rank`` functionality and return format, checking that:
            - input classifier instance is not utilized further than getting
              its configuration.
            - methods are invoked on re-instanced classifier appropriately with
              expected ``build_index`` products.
            - returned dictionary format is appropriate
        """

        class DummyClassifier (SupervisedClassifier):
            """ Mock supervised classifier to track type usage. """
            @classmethod
            def is_usable(cls):
                return True

            def get_config(self):
                return {}

            has_model = mock.MagicMock()
            get_labels = mock.MagicMock()
            _train = mock.MagicMock()

            # Include dummy return for the label that should have been trained
            # with
            _classify_arrays = None

            # Also mocking surface methods to bypass super-class functionality
            # for the purposes of this test.
            train = mock.MagicMock()
            classify_arrays = mock.MagicMock(
                side_effect=lambda v_iter: ({'pos': 1., 'neg': 0.2}
                                            for _ in v_iter)
            )

        c_inst = DummyClassifier()
        # For tracking that this input instance is not functionally used beyond
        # the ``get_config`` method.
        c_inst.get_config = mock.MagicMock(
            side_effect=functools.partial(DummyClassifier.get_config, c_inst)
        )
        c_inst.train = mock.MagicMock()
        c_inst.classify_arrays = mock.MagicMock()

        ri = SupervisedClassifierRelevancyIndex(c_inst)

        # Mock building index
        # - Not just using mock objects to test output format.
        m_mat = list(range(10))
        ri._descr_matrix = m_mat
        # Replacing elements with strings for simplicity. Instance should not
        # be doing anything with elements anyway besides satisfying formatting
        # (ugh).
        m_elems = [str(v) for v in m_mat]
        ri._descr_elem_list = m_elems

        # Invoke ``rank`` with mock input
        m_pos = mock.MagicMock()
        m_neg = mock.MagicMock()
        ret = ri.rank(m_pos, m_neg)

        # Check expected output dictionary format, given than elements are
        # strings for the sake if this test.
        expected_ret = {e: 1.0 for e in m_elems}
        assert ret == expected_ret

        # Check that input classifier only had ``get_config`` called on it,
        # and not ``train`` nor ``classify_arrays``.
        c_inst.get_config.assert_called_once_with()
        c_inst.train.assert_not_called()
        c_inst.classify_arrays.assert_not_called()

        # DummyClassifier class-level mocks effectively track what is called
        # within ``rank``.
        DummyClassifier.train.assert_called_once_with({
            'pos': m_pos, 'neg': m_neg
        })
        DummyClassifier.classify_arrays \
                       .assert_called_once_with(ri._descr_matrix)
