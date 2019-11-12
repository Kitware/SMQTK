from __future__ import division, print_function

import pickle
import unittest

import mock
import pytest

from smqtk.exceptions import NoClassificationError
from smqtk.representation import ClassificationElement


class DummyCEImpl (ClassificationElement):

    @classmethod
    def is_usable(cls):
        # Required to be True to construct.
        return True

    def __getstate__(self):
        # Pass through base-class state for testing.
        return super(DummyCEImpl, self).__getstate__()

    def __setstate__(self, state):
        # pass state through to base-class for testing.
        super(DummyCEImpl, self).__setstate__(state)

    def get_config(self):
        raise NotImplementedError()

    def get_classification(self):
        raise NotImplementedError()

    def set_classification(self, m=None, **kwds):
        raise NotImplementedError()

    def has_classifications(self):
        raise NotImplementedError()


class TestClassificationElementAbstract (unittest.TestCase):

    def test_init(self):
        e = DummyCEImpl('foo', 'bar')
        self.assertEqual(e.type_name, 'foo')
        self.assertEqual(e.uuid, 'bar')

    def test_hash(self):
        self.assertEqual(hash(DummyCEImpl('foo', 'bar')),
                         hash(('foo', 'bar')))

    def test_equality(self):
        """
        Test that two classification elements that return the same non-empty
        dictionary from ``get_classification`` are considered equal.
        """
        expected_map = {'a': 0.2, 'b': 0.3, 0: 0.5}

        # noinspection PyTypeChecker
        inst1 = DummyCEImpl(None, None)
        # noinspection PyTypeChecker
        inst2 = DummyCEImpl(None, None)

        # Mock classification map return for both instances to a valid dict.
        inst1.get_classification = mock.MagicMock(return_value=expected_map)
        inst2.get_classification = mock.MagicMock(return_value=expected_map)

        assert inst1 == inst2
        assert not (inst1 != inst2)  # lgtm[py/redundant-comparison]

    def test_equality_other_not_element(self):
        """
        Test that equality fails when other value is not a
        ClassificationElement.
        """
        # noinspection PyTypeChecker
        inst1 = DummyCEImpl(None, None)
        assert not (inst1 == 'foobar')
        assert inst1 != 'foobar'

    def test_equality_self_no_classification(self):
        """
        Test that equality fails when first instance has no classification map.
        """
        # noinspection PyTypeChecker
        inst1 = DummyCEImpl(None, None)
        # noinspection PyTypeChecker
        inst2 = DummyCEImpl(None, None)

        # One inst raises exception, other has valid dict.
        inst1.get_classification = mock.Mock(side_effect=NoClassificationError)
        inst2.get_classification = mock.Mock(return_value={'a': 0.0, 'b': 1.0})

        assert not (inst1 == inst2)
        assert inst1 != inst2  # lgtm[py/redundant-comparison]

    def test_equality_other_no_classification(self):
        """
        Test that equality fails when second instance has no classification
        map.
        """
        # noinspection PyTypeChecker
        inst1 = DummyCEImpl(None, None)
        # noinspection PyTypeChecker
        inst2 = DummyCEImpl(None, None)

        # One inst raises exception, other has valid dict.
        inst1.get_classification = mock.Mock(return_value={'a': 0.0, 'b': 1.0})
        inst2.get_classification = mock.Mock(side_effect=NoClassificationError)

        assert not (inst1 == inst2)
        assert inst1 != inst2  # lgtm[py/redundant-comparison]

    def test_equality_both_no_classification(self):
        """
        Test that two elements are euqal when they both do not have
        classification maps set.
        """
        # noinspection PyTypeChecker
        inst1 = DummyCEImpl(None, None)
        # noinspection PyTypeChecker
        inst2 = DummyCEImpl(None, None)

        # Mock both instances to raise exception.
        inst1.get_classification = mock.Mock(side_effect=NoClassificationError)
        inst2.get_classification = mock.Mock(side_effect=NoClassificationError)

        assert inst1 == inst2
        assert not (inst1 != inst2)  # lgtm[py/redundant-comparison]

    def test_get_items(self):
        e1 = DummyCEImpl('test', 0)
        e1.get_classification = mock.Mock(return_value={1: 1, 2: 0})

        # There should not be a KeyError for accessing the test labels, but
        # there should for a different label that is not includes in the element
        self.assertEqual(e1[1], 1)
        self.assertEqual(e1[2], 0)
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            e1['some other key']

    def test_bool_cast(self):
        """
        Testing that boolean conversion matches has_classifications
        return.
        """
        e = DummyCEImpl('test', 0)

        expected = False
        e.has_classifications = mock.Mock(return_value=expected)
        assert bool(e) is expected, "Element did not cast to False when " \
                                    "has_classifications returned False."

        expected = True
        e.has_classifications = mock.Mock(return_value=expected)
        assert bool(e) is expected, "Element did not cast to True when " \
                                    "has_classifications returned True."

    def test_getstate(self):
        """
        Test the expected state representation of the abstract base class.
        """
        expected_typename = 'test type name'
        expected_uuid = ' test uuid;'
        expected_state = (expected_typename, expected_uuid)

        #: :type: ClassificationElement
        inst = mock.MagicMock(spec_set=ClassificationElement)
        inst.type_name = expected_typename
        inst.uuid = expected_uuid
        actual_state = ClassificationElement.__getstate__(inst)

        assert actual_state == expected_state

    def test_setstate(self):
        """
        Test that the expected state representation correctly sets
        :return:
        """
        expected_typename = 'test type name'
        expected_uuid = ' test uuid;'
        given_state = (expected_typename, expected_uuid)

        #: :type: ClassificationElement
        inst = mock.MagicMock(spec_set=ClassificationElement)
        ClassificationElement.__setstate__(inst, given_state)

        assert inst.type_name == expected_typename
        assert inst.uuid == expected_uuid

    def test_serialize_deserialize_pickle(self):
        """
        Test that we can serialize and deserialize this abstract base class
        component.
        """
        expected_typename = 'test type name'
        expected_uuid = ' test uuid;'

        inst1 = DummyCEImpl(expected_typename, expected_uuid)
        buff = pickle.dumps(inst1)
        #: :type: ClassificationElement
        inst2 = pickle.loads(buff)
        # Intentionally checking not the same instance
        assert inst2 is not inst1  # lgtm[py/comparison-using-is]
        assert inst2.type_name == expected_typename
        assert inst2.uuid == expected_uuid

    def test_max_label_no_classification_error(self):
        """
        Test that NoClassificationError is raised when we attemp to get the
        classification map with none set.
        """
        # Mock classification element instance
        e = mock.MagicMock(spec_set=ClassificationElement)
        e.get_classification.return_value = {}

        self.assertRaises(
            NoClassificationError,
            ClassificationElement.max_label, e
        )

    def test_max_label_zero_confidence(self):
        """
        Test that if there *are* classifications but the maximum conf is 0 that
        at least one label is returned.
        """
        # Mock classification element instance
        e = mock.MagicMock(spec_set=ClassificationElement)
        e.get_classification.return_value = {'a': 0.0}
        ClassificationElement.max_label(e)

    def test_get_default_config(self):
        """
        Test that the default configuration does not include the runtime
        specific parameters.
        """
        # Shows that override in ClassificationElement removes those
        # runtime-specific parameters.
        default = ClassificationElement.get_default_config()
        assert 'type_name' not in default
        assert 'uuid' not in default

    @mock.patch('smqtk.utils.configuration.Configurable.from_config')
    def test_from_config_mdFalse(self, m_confFromConfig):
        """
        Test that ``from_config`` appropriately passes runtime provided
        parameters.
        """
        given_conf = {}
        expected_typename = 'ex typename'
        expected_uuid = 'ex uuid'
        expected_conf = {
            'type_name': expected_typename,
            'uuid': expected_uuid,
        }
        expected_return = 'sim return from parent'

        m_confFromConfig.return_value = expected_return

        r = ClassificationElement.from_config(given_conf, expected_typename,
                                              expected_uuid,
                                              merge_default=False)

        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=False)
        assert r == expected_return

    @mock.patch('smqtk.utils.configuration.Configurable.from_config')
    def test_from_config_mdTrue(self, m_confFromConfig):
        """
        Test that ``from_config`` appropriately passes runtime provided
        parameters.
        """
        given_conf = {}
        expected_typename = 'ex typename'
        expected_uuid = 'ex uuid'
        expected_conf = {
            'type_name': expected_typename,
            'uuid': expected_uuid,
        }
        expected_return = 'sim return from parent'

        m_confFromConfig.return_value = expected_return

        r = ClassificationElement.from_config(given_conf, expected_typename,
                                              expected_uuid,
                                              merge_default=True)

        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=True)
        assert r == expected_return

    @mock.patch('smqtk.utils.configuration.Configurable.from_config')
    def test_from_config_preseeded_mdFalse(self, m_confFromConfig):
        """
        Test that parameters provided at runtime prevails over any provided
        through a given config.
        """
        given_conf = {
            "type_name": "should not get through",
            "uuid": "should not get through",
        }
        expected_typename = 'actually expected typename'
        expected_uuid = 'actually expected uuid'
        expected_conf = {
            'type_name': expected_typename,
            'uuid': expected_uuid,
        }
        expected_return = 'sim return from parent'
        m_confFromConfig.return_value = expected_return

        r = ClassificationElement.from_config(given_conf, expected_typename,
                                              expected_uuid,
                                              merge_default=False)

        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=False)
        assert r == expected_return

    @mock.patch('smqtk.utils.configuration.Configurable.from_config')
    def test_from_config_preseeded_mdTrue(self, m_confFromConfig):
        """
        Test that parameters provided at runtime prevails over any provided
        through a given config.
        """
        given_conf = {
            "type_name": "should not get through",
            "uuid": "should not get through",
        }
        expected_typename = 'actually expected typename'
        expected_uuid = 'actually expected uuid'
        expected_conf = {
            'type_name': expected_typename,
            'uuid': expected_uuid,
        }
        expected_return = 'sim return from parent'
        m_confFromConfig.return_value = expected_return

        r = ClassificationElement.from_config(given_conf, expected_typename,
                                              expected_uuid,
                                              merge_default=True)

        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=True)
        assert r == expected_return

    def test_max_label(self):
        """
        Test that max_label correctly returns the label key associated with
        the greatest associated confidence value.
        """
        # Mock classification element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)
        e.get_classification.return_value = {1: 0, 2: 1, 3: 0.5}

        expected_label = 2
        actual_label = ClassificationElement.max_label(e)
        self.assertEqual(actual_label, expected_label)

    def test_set_no_input(self):
        """
        Test that calling ``set_classification`` with default arguments raises
        a ValueError.
        """
        # Mock element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)

        with pytest.raises(ValueError, match="No classification labels/values "
                                             "given\."):
            ClassificationElement.set_classification(e)

    def test_set_empty_input(self):
        """
        Test that calling ``set_classification`` with an empty dictionary
        raises a ValueError.
        """
        # Mock element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)

        with pytest.raises(ValueError, match="No classification labels/values "
                                             "given\."):
            ClassificationElement.set_classification(e, {})

    def test_set_input_dict(self):
        """
        Test that passing a dictionary to ``set_classification`` returns the
        appropriately normalized dictionary.
        """
        # Mock element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)

        expected_v = {1: 0, 2: 1}
        actual_v = ClassificationElement.set_classification(e, expected_v)
        assert actual_v == expected_v

    def test_set_kwargs(self):
        """
        Test that passing a keyword arguments to ``set_classification`` returns
        the appropriately normalized dictionary.
        """
        # Mock element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)

        expected_v = {'a': 1, 'b': 0}
        actual_v = ClassificationElement.set_classification(e, a=1, b=0)
        assert actual_v == expected_v

    def test_set_mixed(self):
        """
        Test that passing mixed dictionary and keyword arguments to
        ``set_classification`` returns the appropriately normalize dictionary.
        """
        # Mock element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)

        expected_v = {'a': .25, 1: .25, 'b': .25, 'd': .25}
        actual_v = ClassificationElement.set_classification(e,
                                                            {'a': .25, 1: .25},
                                                            b=.25, d=.25)
        assert actual_v == expected_v

    def test_set_nonstandard(self):
        """
        Test that setting a set of label/confidence pairs where the
        confidence sums to greater than 1.0 is acceptable and returns the
        appropriately normalized dictionary.

        Many classifiers output 1-sum confidence values, but not all (e.g.
        CNN final layers like AlexNet).
        """
        # Mock element instance
        #: :type: ClassificationElement
        e = mock.MagicMock(spec_set=ClassificationElement)

        expected_v = {'a': 1, 1: 1, 'b': 1, 'd': 1}
        actual_v = ClassificationElement.set_classification(e, {'a': 1, 1: 1},
                                                            b=1, d=1)
        assert actual_v == expected_v
