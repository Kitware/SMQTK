from __future__ import division, print_function
import unittest
import unittest.mock as mock

import numpy as np
import pytest

from smqtk.algorithms.classifier import Classifier
from smqtk.representation import (
    ClassificationElement,
    ClassificationElementFactory,
    DescriptorElement,
)
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


class DummyClassifier (Classifier):

    EXPECTED_LABELS = ['constant']

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def get_labels(self):
        return self.EXPECTED_LABELS

    def _classify_arrays(self, array_iter):
        """
        Some deterministic dummy impl
        Simply returns a classification with one label "test" whose value is
        the first value of the vector
        """
        for v in array_iter:
            yield {'test': v[0]}
        self._post_iterator_check()

    def _post_iterator_check(self):
        """ Stub method for testing functionality is called post-final-yield.
        """

    def _classify_too_few(self, array_iter):
        """ Swap-in for _classify_arrays that under-generates."""
        # Yield all but one
        array_list = list(array_iter)
        for i, v in enumerate(array_list[:-1]):
            yield {'test': i}
        self._post_iterator_check()

    def _classify_too_many(self, array_iter):
        """ Swap-in for _classify_arrays that over-generates."""
        i = 0
        for i, v in enumerate(array_iter):
            yield {'test': i}
        # Yield some extra stuff
        yield {'test': i+1}
        yield {'test': i+2}
        self._post_iterator_check()


class TestClassifierAbstractClass (unittest.TestCase):

    def setUp(self):
        # Common dummy instance setup per test case.
        self.inst = DummyClassifier()
        self.inst._post_iterator_check = mock.Mock()

    def test_classify_arrays_inconsistent(self):
        """ Test that passing arrays of inconsistent dimensionality causes a
        ValueError.
        """
        arrs = list(map(np.array, [[1, 2, 3], [1, 2], [1, 2, 3, 4]]))
        with pytest.raises(ValueError):
            list(self.inst.classify_arrays(arrs))

    def test_classify_arrays_empty_iter(self):
        """ Test that passing an empty iterator correctly yields another empty
        iterator."""
        arrs = []
        assert list(self.inst.classify_arrays(arrs)) == []
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_arrays(self):
        """ Test "successful" function of classify arrays. """
        arrs = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        list(self.inst.classify_arrays(arrs))
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_elements_empty_iter(self):
        """ Test that passing an empty iterator to classify elements returns
        an empty iterator.

        ``iter_tocompute_arrays`` would yield nothing of course, setting EoI to
        -1 since there was nothing yielded.
        """
        elems = []
        assert list(self.inst.classify_elements(elems)) == []
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_elements_inconsistent(self):
        """ Test that elements with inconsistent vector dims raise same
        ValueError as the ``test_classify_arrays_inconsistent`` test."""
        arrs = list(map(np.array, [[1, 2, 3], [1, 2, 4], [1, 2, 3, 4]]))
        elems = [
            DescriptorMemoryElement('', i).set_vector(v)
            for i, v in enumerate(arrs)
        ]
        with pytest.raises(ValueError, match=r"violated dimension consistency"):
            list(self.inst.classify_elements(elems))

    def test_classify_elements_missing_vector(self):
        """ Test that we get a ValueError when """
        elems = [
            DescriptorMemoryElement('', 0).set_vector([1, 2, 3]),
            DescriptorMemoryElement('', 0),  # no set vector
            DescriptorMemoryElement('', 0).set_vector([4, 5, 6]),
        ]
        with pytest.raises(ValueError, match=r"no vector stored"):
            list(self.inst.classify_elements(elems))

    def test_classify_elements_impl_under_generates(self):
        """ Test that we catch when an implementation under generates
        classifications relative to input descriptors."""
        arrs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        elems = [
            DescriptorMemoryElement('', i).set_vector(v)
            for i, v in enumerate(arrs)
        ]

        # Switch inst._classify_arrays to one that under produces
        self.inst._classify_arrays = self.inst._classify_too_few

        with pytest.raises(IndexError, match=r"under-produced classifications"):
            list(self.inst.classify_elements(elems))

    def test_classify_elements_impl_over_generates(self):
        """ Test that we catch when an implementation over generates
        classifications relative to input descriptors."""
        elems = [
            DescriptorMemoryElement('', i).set_vector(v)
            for i, v in enumerate([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]])
        ]

        # Switch inst._classify_arrays to one that over produces
        self.inst._classify_arrays = self.inst._classify_too_many

        with pytest.raises(IndexError, match=r"over-produced classifications"):
            list(self.inst.classify_elements(elems))

    def test_classify_elements_none_preexisting(self):
        """ Test generating classification elements where none generated by the
        factory have existing vectors. i.e. all descriptor elements passed to
        underlying classification method."""
        d_elems = [
            DescriptorMemoryElement('', i).set_vector(v)
            for i, v in enumerate([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]])
        ]

        # Mock a factory to produce elements whose ``has_classifications``
        # method returns False.
        m_ce_type = mock.MagicMock(name="MockedClassificationElementType")
        c_factory = ClassificationElementFactory(
            m_ce_type, {}
        )
        # Mocking that elements have no classifications set
        m_ce_inst = m_ce_type.from_config()
        m_ce_inst.has_classifications.return_value = False

        list(self.inst.classify_elements(d_elems, factory=c_factory,
                                         overwrite=False))
        assert m_ce_inst.has_classifications.call_count == 3
        assert m_ce_inst.set_classification.call_count == 3
        # Check that expected classification returns from dummy generator were
        # set to factory-created elements.
        m_ce_inst.set_classification.assert_any_call({'test': 1})
        m_ce_inst.set_classification.assert_any_call({'test': 4})
        m_ce_inst.set_classification.assert_any_call({'test': 7})

        # Dummy classifier iterator completed to the end.
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_elements_all_preexisting(self):
        """ Test generating classification elements where all elements
        generated by the factory claim to already have classifications and
        overwrite is False."""
        d_elems = [
            DescriptorMemoryElement('', i).set_vector(v)
            for i, v in enumerate([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]])
        ]

        # Mock a factory to produce elements whose ``has_classifications``
        # method returns False.
        m_ce_type = mock.MagicMock(name="MockedClassificationElementType")
        c_factory = ClassificationElementFactory(
            m_ce_type, {}
        )
        # Mocking that elements have no classifications set
        m_ce_inst = m_ce_type.from_config()
        m_ce_inst.has_classifications.return_value = True

        list(self.inst.classify_elements(d_elems, factory=c_factory,
                                         overwrite=False))
        assert m_ce_inst.has_classifications.call_count == 3
        m_ce_inst.set_classification.assert_not_called()

        # Dummy classifier iterator completed to the end.
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_elements_all_preexisting_overwrite(self):
        """ Test generating classification elements where all elements
        generated by the factory claim to already have classifications but
        overwrite is True this time."""
        d_elems = [
            DescriptorMemoryElement('', i).set_vector(v)
            for i, v in enumerate([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]])
        ]

        # Mock a factory to produce elements whose ``has_classifications``
        # method returns False.
        m_ce_type = mock.MagicMock(name="MockedClassificationElementType")
        c_factory = ClassificationElementFactory(
            m_ce_type, {}
        )
        # Mocking that elements have no classifications set
        m_ce_inst = m_ce_type.from_config()
        m_ce_inst.has_classifications.return_value = True

        list(self.inst.classify_elements(d_elems, factory=c_factory,
                                         overwrite=True))
        # Method not called becuase of overwrite short-circuit
        assert m_ce_inst.has_classifications.call_count == 0
        assert m_ce_inst.set_classification.call_count == 3
        # Check that expected classification returns from dummy generator were
        # set to factory-created elements.
        m_ce_inst.set_classification.assert_any_call({'test': 1})
        m_ce_inst.set_classification.assert_any_call({'test': 4})
        m_ce_inst.set_classification.assert_any_call({'test': 7})

        # Dummy classifier iterator completed to the end.
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_elements_mixed_precomp(self):
        """ Test that a setup with mixed pre-computed and not
        ClassificationElements from the factory results in all elements yielded
        correctly."""
        # Setup descriptor elements and paired classification elements the
        # mock factory will be producing.
        descr_elems = []
        exp_c_elems = []
        for i in range(8):
            de = DescriptorMemoryElement('', i).set_vector([i])
            descr_elems.append(de)

            ce = mock.MagicMock(spec=ClassificationElement)
            # Make some elements not "have classifications"
            if i in [2, 5, 6]:
                ce.has_classifications.return_value = False
            else:
                ce.has_classifications.return_value = True
            exp_c_elems.append(ce)

        def m_fact_new_desc(_, uid):
            # UIDs aligned with integer index.
            return exp_c_elems[uid]

        m_fact = mock.MagicMock(spec=ClassificationElementFactory)
        m_fact.new_classification.side_effect = m_fact_new_desc

        act_c_elems = list(self.inst.classify_elements(
            descr_elems, factory=m_fact, overwrite=False,
        ))
        assert act_c_elems == exp_c_elems

        # Check that the expected checks and sets occurred
        # - All classification elements should have been checked for
        #   ``has_classifications``
        for e in act_c_elems:
            e.has_classifications.assert_called_once()
        # - Check that set calls were appropriately called or not called.
        act_c_elems[0].set_classification.assert_not_called()
        act_c_elems[1].set_classification.assert_not_called()
        act_c_elems[2].set_classification.assert_called_once_with({'test': 2})
        act_c_elems[3].set_classification.assert_not_called()
        act_c_elems[4].set_classification.assert_not_called()
        act_c_elems[5].set_classification.assert_called_once_with({'test': 5})
        act_c_elems[6].set_classification.assert_called_once_with({'test': 6})
        act_c_elems[7].set_classification.assert_not_called()

        # Dummy classifier iterator completed to the end.
        self.inst._post_iterator_check.assert_called_once()

    def test_classify_elements_batching_effect(self):
        """ Test that setting different values to the descriptor fetch batching
        parameter has an effect.
        """
        # Check that the DescritporElement.get_many_vectors class method is
        # invoked with differently sized lists based on the input d_elem_batch
        # parameter.

        d_elems = [
            DescriptorMemoryElement('', i).set_vector([i])
            for i in range(29)
        ]

        # Setup a mock factory that returns expected objects for
        exp_ce_list = [
            mock.MagicMock(spec=ClassificationElement)
            for _ in range(29)
        ]
        dummy_fact = mock.Mock(spec=ClassificationElementFactory)
        dummy_fact.new_classification.side_effect = \
            lambda _, uid: exp_ce_list[uid]

        # batch default of 100 == 1 call
        with mock.patch('smqtk.algorithms.classifier._interface_classifier.'
                        'DescriptorElement.get_many_vectors',
                        wraps=DescriptorMemoryElement.get_many_vectors)\
                as m_DE_gmv:
            act_ce_list = list(self.inst.classify_elements(d_elems,
                                                           factory=dummy_fact))
            assert act_ce_list == exp_ce_list
            m_DE_gmv.assert_called_once()
            m_DE_gmv.assert_called_with(d_elems)

        # batch of 1 == 29 calls
        with mock.patch('smqtk.algorithms.classifier._interface_classifier.'
                        'DescriptorElement.get_many_vectors',
                        wraps=DescriptorMemoryElement.get_many_vectors)\
                as m_DE_gmv:
            act_ce_list = list(self.inst.classify_elements(d_elems,
                                                           factory=dummy_fact,
                                                           d_elem_batch=1))
            assert act_ce_list == exp_ce_list
            assert m_DE_gmv.call_count == 29
            for de in d_elems:
                m_DE_gmv.assert_any_call([de])

        # batch of 20 == 2 calls
        with mock.patch('smqtk.algorithms.classifier._interface_classifier.'
                        'DescriptorElement.get_many_vectors',
                        wraps=DescriptorMemoryElement.get_many_vectors) \
                as m_DE_gmv:
            act_ce_list = list(self.inst.classify_elements(d_elems,
                                                           factory=dummy_fact,
                                                           d_elem_batch=20))
            assert act_ce_list == exp_ce_list
            assert m_DE_gmv.call_count == 2
            m_DE_gmv.assert_any_call(d_elems[:20])
            m_DE_gmv.assert_any_call(d_elems[20:])
