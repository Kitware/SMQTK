import functools
from unittest import mock

import numpy as np

from smqtk.algorithms.rank_relevancy.wrap_classifier import (
    RankRelevancyWithSupervisedClassifier,
)
from smqtk.algorithms import SupervisedClassifier


def test_is_usable():
    """ test that this impl is always available. """
    assert RankRelevancyWithSupervisedClassifier.is_usable()


def test_rank():
    """
    Test wrapper ``rank`` functionality and return format, checking that:
        - input classifier instance is not utilized further than getting
          its configuration.
        - methods are invoked on re-instanced classifier appropriately.
        - returned sequence format is appropriate
    """

    class DummyClassifier (SupervisedClassifier):
        """ Mock supervised classifier to track type usage. """
        def get_config(self):
            return {}

        # Set unused abstract methods to None to allow class
        # instantiation
        has_model = None
        get_labels = None
        _train = None
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

    rr = RankRelevancyWithSupervisedClassifier(c_inst)

    # Not just using mock objects to test output format.
    m_mat = np.arange(10)
    # Invoke ``rank`` with mock input
    m_pos = mock.MagicMock()
    m_neg = mock.MagicMock()
    ret = rr.rank(m_pos, m_neg, m_mat)

    # Check expected output sequence format.
    assert len(ret) == len(m_mat) and all(x == 1.0 for x in ret)

    # Check that input classifier only had ``get_config`` called on it,
    # and not ``train`` nor ``classify_arrays``.
    c_inst.get_config.assert_called_once_with()
    c_inst.train.assert_not_called()
    c_inst.classify_arrays.assert_not_called()

    # DummyClassifier class-level mocks effectively track what is called
    # within ``rank``.
    DummyClassifier.train.assert_called_once()
    tca_args, tca_kwargs = DummyClassifier.train.call_args
    assert len(tca_args) == 1 and not tca_kwargs
    # Two classes, positive and negative
    assert len(tca_args[0]) == 2
    DummyClassifier.classify_arrays.assert_called_once()
    caca_args, caca_kwargs = DummyClassifier.classify_arrays.call_args
    assert len(caca_args) == 1 and not caca_kwargs
    assert np.array_equal(caca_args[0], m_mat)
