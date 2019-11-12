import mock
import pytest

from smqtk.exceptions import NoDetectionError
from smqtk.representation import DetectionElement


###############################################################################
# Helper classes and methods

class DummyDetectionElement (DetectionElement):
    """
    Dummy implementation for testing methods implemented in abstract parent
    class (no constructor override). Abstract methods are not implemented
    beyond declaration.
    """

    # Satisfy Pluggable ##################################

    @classmethod
    def is_usable(cls):
        return True

    # Satisfy Configurable ###############################

    def get_config(self):
        raise NotImplementedError()

    # Satisfy DetectionElement ###########################

    def __getstate__(self):
        raise NotImplementedError()

    def __setstate__(self, state):
        raise NotImplementedError()

    def has_detection(self):
        raise NotImplementedError()

    def set_detection(self, bbox, classification_element):
        raise NotImplementedError()

    def get_detection(self):
        raise NotImplementedError()


###############################################################################
# Tests

def test_construction():
    """
    Test that normal construction sets the correct attributes
    """
    expected_uuid = 0
    m = mock.MagicMock(spec_set=DetectionElement)
    # noinspection PyCallByClass
    DetectionElement.__init__(m, expected_uuid)
    assert m._uuid == expected_uuid


def test_get_default_config_override():
    """
    Test override of get_default_config s.t. ``uuid`` is not present in the
    result dict.
    """
    default = DetectionElement.get_default_config()
    assert 'uuid' not in default


@mock.patch('smqtk.utils.configuration.Configurable.from_config')
def test_from_config_override_mdFalse(m_confFromConfig):
    """
    Test that ``from_config`` appropriately passes runtime-provided UUID value.
    """
    given_conf = {}
    expected_uuid = 'test uuid'
    expected_conf = {
        'uuid': expected_uuid
    }

    DetectionElement.from_config(given_conf, expected_uuid,
                                 merge_default=False)
    m_confFromConfig.assert_called_once_with(expected_conf,
                                             merge_default=False)


@mock.patch('smqtk.utils.configuration.Configurable.from_config')
def test_from_config_override_mdTrue(m_confFromConfig):
    """
    Test that ``from_config`` appropriately passes runtime-provided UUID value.
    """
    given_conf = {}
    expected_uuid = 'test uuid'
    expected_conf = {
        'uuid': expected_uuid
    }

    DetectionElement.from_config(given_conf, expected_uuid,
                                 merge_default=True)
    m_confFromConfig.assert_called_once_with(expected_conf,
                                             merge_default=False)


@mock.patch('smqtk.utils.configuration.Configurable.from_config')
def test_from_config_uuid_preseed_mdFalse(m_confFromConfig):
    """
    Test that UUID provided at runtime prevails over any UUID provided
    through the config.
    """
    given_conf = {
        "uuid": "should not get through",
    }
    expected_uuid = "actually expected UUID"
    expected_conf = {
        'uuid': expected_uuid
    }

    DetectionElement.from_config(given_conf, expected_uuid,
                                 merge_default=False)
    m_confFromConfig.assert_called_once_with(expected_conf,
                                             merge_default=False)


@mock.patch('smqtk.utils.configuration.Configurable.from_config')
def test_from_config_uuid_preseed_mdTrue(m_confFromConfig):
    """
    Test that UUID provided at runtime prevails over any UUID provided
    through the config.
    """
    given_conf = {
        "uuid": "should not get through",
    }
    expected_uuid = "actually expected UUID"
    expected_conf = {
        'uuid': expected_uuid
    }

    DetectionElement.from_config(given_conf, expected_uuid,
                                 merge_default=True)
    m_confFromConfig.assert_called_once_with(expected_conf,
                                             merge_default=False)


def test_hash():
    """
    Test that a DetectionElement is hashable based on solely on UUID.
    """
    with pytest.raises(TypeError, match="unhashable type"):
        hash(DummyDetectionElement(0))


def test_eq_both_no_detections():
    """
    Test that two elements with no detection info set are considered not equal.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)
    d1.get_detection = d2.get_detection = \
        mock.MagicMock(side_effect=NoDetectionError)
    assert (d1 == d2) is False
    assert (d2 == d1) is False
    assert (d1 != d2) is True
    assert (d2 != d1) is True


def test_eq_one_no_detection():
    """
    Test that when one element has no detection info then they are considered
    NOT equal.
    """
    d_without = DummyDetectionElement(0)
    d_without.get_detection = mock.MagicMock(side_effect=NoDetectionError)
    d_with = DummyDetectionElement(1)
    d_with.get_detection = mock.MagicMock(return_value=(1, 2))

    assert (d_with == d_without) is False
    assert (d_without == d_with) is False
    assert (d_with != d_without) is True
    assert (d_without != d_with) is True


def test_eq_unequal_detections():
    """
    Test that two detections, with valid, but different contents, test out not
    equal.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)
    d1.get_detection = mock.Mock(return_value=('a', 1))
    d2.get_detection = mock.Mock(return_value=('b', 2))
    assert (d1 == d2) is False


def test_eq_unequal_just_one():
    """
    Test inequality where just one of the two sub-components of detections (bb,
    classification) are different.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)

    d1.get_detection = mock.Mock(return_value=('a', 1))
    d2.get_detection = mock.Mock(return_value=('a', 2))
    assert (d1 == d2) is False

    d1.get_detection = mock.Mock(return_value=('a', 1))
    d2.get_detection = mock.Mock(return_value=('b', 1))
    assert (d1 == d2) is False


def test_eq_success():
    """
    Test when two different detection instances returns the same value pair
    from ``get_detection()``.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)
    d1.get_detection = d2.get_detection = \
        mock.MagicMock(return_value=('a', 0))
    assert d1 == d2


def test_nonzero_has_detection():
    """
    Test that boolean cast of a DetectionElement occurs appropriately when the
    element has a detection.
    """
    expected_val = True
    inst = DummyDetectionElement(0)
    inst.has_detection = mock.MagicMock(return_value=expected_val)
    assert bool(inst) is expected_val
    inst.has_detection.assert_called_once_with()


def test_nonzero_no_detection():
    """
    Test that boolean cast of a DetectionElement occurs appropriately when the
    element has a detection.
    """
    expected_val = False
    inst = DummyDetectionElement(0)
    inst.has_detection = mock.MagicMock(return_value=expected_val)
    assert bool(inst) is expected_val
    inst.has_detection.assert_called_once_with()


def test_property_uuid():
    """
    Test that given UUID hashable is returned via `uuid` property.
    """
    expected_uuid = 0
    assert DummyDetectionElement(expected_uuid).uuid == expected_uuid

    expected_uuid = 'a hashable string'
    assert DummyDetectionElement(expected_uuid).uuid == expected_uuid


def test_getstate():
    """
    Test that expected "state" representation is returned from __getstate__.
    """
    expected_uuid = 'expected-uuid'
    expected_state = {
        '_uuid': expected_uuid
    }

    # Mock an instance of DetectionElement with expected uuid attribute set.
    m = mock.MagicMock(spec_set=DetectionElement)
    m._uuid = expected_uuid

    actual_state = DetectionElement.__getstate__(m)
    assert actual_state == expected_state


def test_setstate():
    """
    Test that __setstate__ base implementation sets the correct instance
    attributes.
    """
    expected_uuid = 'expected_uuid'
    expected_state = {
        '_uuid': expected_uuid
    }

    # Mock an instance of DetectionElement
    m = mock.MagicMock(spec_set=DetectionElement)

    # noinspection PyCallByClass
    # - for testing purposes.
    DetectionElement.__setstate__(m, expected_state)
    assert m._uuid == expected_uuid
