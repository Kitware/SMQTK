import mock
import pytest

from smqtk.representation import DetectionElement


###############################################################################
# Helper classes and methods

class DummyDetectionElement (DetectionElement):
    """
    Dummy implementation for testing abstract methods of DataElement.
    """

    # Satisfy Pluggable ##################################

    @classmethod
    def is_usable(cls):
        return True

    # Satisfy Configurable ###############################

    def get_config(self):
        return {}

    # Satisfy DetectionElement ###########################

    def has_detection(self):
        pass

    def set_detection(self, bbox, classification_element):
        pass

    def get_detection(self):
        pass


###############################################################################
# Tests

def test_detection_element_construction():
    """
    Test that normal construction sets the correct attributes
    """
    DummyDetectionElement(0)


def test_detection_element_hash():
    """
    Test that a DetectionElement is not hashable.
    """
    assert hash(DummyDetectionElement(0)) == hash(0)
    assert hash(DummyDetectionElement('some-str')) == hash('some-str')


def test_nonzero_has_detection():
    """
    Test that boolean cast of a DetectionElement occurs appropriately when the
    element has a detection.
    """
    expected_val = True
    inst = DummyDetectionElement(0)
    inst.has_detection = mock.MagicMock(return_value=expected_val)
    assert bool(inst) is expected_val


def test_nonzero_no_detection():
    """
    Test that boolean cast of a DetectionElement occurs appropriately when the
    element has a detection.
    """
    expected_val = False
    inst = DummyDetectionElement(0)
    inst.has_detection = mock.MagicMock(return_value=expected_val)
    assert bool(inst) is expected_val


def test_property_uuid():
    """
    Test that given UUID hashable is returned via `uuid` property.
    """
    expected_uuid = 0
    assert DummyDetectionElement(expected_uuid).uuid == expected_uuid

    expected_uuid = 'a hashable string'
    assert DummyDetectionElement(expected_uuid).uuid == expected_uuid
