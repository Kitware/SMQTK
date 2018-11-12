import mock
import pytest

from smqtk.exceptions import NoDetectionError
from smqtk.representation import AxisAlignedBoundingBox, ClassificationElement
from smqtk.representation.detection_element.memory \
    import MemoryDetectionElement


def test_is_usable():
    """ Test that memory impl is usable (should always be). """
    assert MemoryDetectionElement.is_usable() is True


def test_get_config():
    """ Test that configuration for memory element is empty. """
    assert MemoryDetectionElement(0).get_config() == {}


def test_has_detection():
    """
    Test that has_detection is true for True-evaluating attributes
    """
    inst = MemoryDetectionElement(0)
    inst._bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    # Simulate having a non-empty element.
    inst._classification = mock.MagicMock(spec=ClassificationElement)
    inst._classification.has_classifications.return_value = True

    assert inst.has_detection() is True


def test_has_detection_none_members():
    """
    Test that has_detection is false when neither bbox nor classification
    are set.
    """
    inst = MemoryDetectionElement(0)
    inst._bbox = inst._classification = None
    assert inst.has_detection() is False


def test_has_detection_one_none_member():
    """
    Test that has_detections is false if at least one of the members is None.
    """
    # Possible "valid" values.
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    celem = mock.MagicMock(spec=ClassificationElement)
    celem.__nonzero__.return_value = celem.__bool__.return_value = True
    celem.has_classifications.return_value = True

    inst = MemoryDetectionElement(0)
    inst._bbox = None
    inst._classification = celem
    assert inst.has_detection() is False

    inst._bbox = bbox
    inst._classification = None
    assert inst.has_detection() is False


def test_has_detection_empty_classification_element():
    """
    Test that when one or both attributes are false-evaluating but not None,
    has_detection returns false.
    """
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    celem = mock.MagicMock(spec=ClassificationElement)
    celem.has_classifications.return_value = False

    inst = MemoryDetectionElement(0)
    inst._bbox = bbox
    inst._classification = celem

    assert inst.has_detection() is False


def test_get_detection():
    """ Test successfully getting the detection components. """
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    c_elem = mock.MagicMock(spec=ClassificationElement)
    # Simulate a populated ClassificationElement
    c_elem.__nonzero__.return_value = c_elem.__bool__.return_value = False

    inst = MemoryDetectionElement(0)
    inst._bbox = bbox
    inst._classification = c_elem
    assert inst.get_detection() == (bbox, c_elem)


def test_get_detection_error_on_empty():
    """
    Test that a NoDetectionError is raised when the detection element has
    not been set to yet.
    """
    inst = MemoryDetectionElement(0)
    with pytest.raises(NoDetectionError,
                       match="Missing detection bounding box or "
                             "missing/invalid classification for in-memory "
                             "detection with UUID 0"):
        inst.get_detection()


def test_get_detection_error_empty_classification():
    """
    Test that NoDetectionError is raised when the classification element is
    false-evaluating.
    """
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    celem = mock.MagicMock(spec=ClassificationElement)
    celem.has_classifications.return_value = False

    inst = MemoryDetectionElement(0)
    inst._bbox = bbox
    inst._classification = celem

    with pytest.raises(NoDetectionError,
                       match="Missing detection bounding box or "
                             "missing/invalid classification for in-memory "
                             "detection with UUID 0"):
        inst.get_detection()


def test_set_detection():
    """
    Test successfully setting a bounding box and classification element.
    """
    # Zero area bbox shouldn't matter, same as a point.
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    c_elem = mock.MagicMock(spec=ClassificationElement)
    # Simulating that c_elem is a valid, populated classification element
    c_elem.__nonzero__.return_value = c_elem.__bool__.return_value = False

    d_elem = MemoryDetectionElement(0)
    # noinspection PyTypeChecker
    d_elem.set_detection(bbox, c_elem)


def test_set_detection_invalid_bbox():
    """
    Test that an exception is raise when a valid bounding box was
    not provided.
    """
    bbox = 'not bbox'
    c_elem = mock.MagicMock(spec=ClassificationElement)
    c_elem.has_classifications.return_value = True

    with pytest.raises(ValueError, match="Provided an invalid "
                                         "AxisAlignedBoundingBox instance. "
                                         "Given 'not bbox' \(type=str\)\."):
        # noinspection PyTypeChecker
        MemoryDetectionElement(0).set_detection(bbox, c_elem)


def test_set_detection_invalid_classification_element():
    """
    Test that an exception is raised when a valid classification element was
    not provided.
    """
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    c_elem = 'not a classification element'

    with pytest.raises(ValueError,
                       match="Provided an invalid ClassificationElement "
                             "instance. Given 'not a classification element' "
                             "\(type=str\)\."):
        # noinspection PyTypeChecker
        MemoryDetectionElement(0).set_detection(bbox, c_elem)


def test_set_detection_empty_classification_element():
    """
    Test that exception is raised when the provided classification element
    is empty (no contents in classification map).
    """
    bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    c_elem = mock.MagicMock(spec=ClassificationElement)
    c_elem.has_classifications.return_value = False

    with pytest.raises(ValueError,
                       match="Given an empty ClassificationElement instance."):
        # noinspection PyTypeChecker
        MemoryDetectionElement(0).set_detection(bbox, c_elem)
