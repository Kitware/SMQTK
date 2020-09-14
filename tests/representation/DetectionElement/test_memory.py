import pickle

import unittest.mock as mock
import pytest

from smqtk.exceptions import NoDetectionError
from smqtk.representation import AxisAlignedBoundingBox, ClassificationElement
from smqtk.representation.classification_element.memory \
    import MemoryClassificationElement
from smqtk.representation.detection_element.memory \
    import MemoryDetectionElement
from smqtk.utils.configuration import configuration_test_helper


def test_is_usable():
    """ Test that memory impl is usable (should always be). """
    assert MemoryDetectionElement.is_usable() is True


def test_serialize_deserialize_pickle():
    """
    Test that serialization and the deserialization of a
    MemoryDetectionElement instance with populated detection components
    results in a different but equivalent instance.
    """
    expected_uuid = 'some-uuid'
    expected_bbox = AxisAlignedBoundingBox([0, 0], [1, 2])
    expected_ce_map = {'a': .24, 'b': 0.5, 0: .26}
    expected_ce = MemoryClassificationElement('ce-type', expected_uuid)
    expected_ce.set_classification(expected_ce_map)

    e1 = MemoryDetectionElement(expected_uuid)
    e1._bbox = expected_bbox
    e1._classification = expected_ce

    buff = pickle.dumps(e1)

    #: :type: MemoryDetectionElement
    e2 = pickle.loads(buff)
    assert e2 is not e1
    assert e2.uuid == expected_uuid

    e2_bbox, e2_ce = e2.get_detection()
    # Intentionally checking e2_bbox is not the same instance
    assert e2_bbox is not expected_bbox  # lgtm[py/comparison-using-is]
    assert e2_bbox == expected_bbox
    assert e2_ce is not expected_ce
    assert e2_ce == expected_ce
    assert e2_ce.get_classification() == expected_ce_map


def test_get_config():
    """ Test that configuration for memory element is empty. """
    inst = MemoryDetectionElement(0)
    for i in configuration_test_helper(inst, {'uuid'}, (0,)):  # type: MemoryDetectionElement
        assert i.uuid == 0


def test_has_detection():
    """
    Test that has_detection is true for True-evaluating attributes
    """
    inst = MemoryDetectionElement(0)
    inst._bbox = mock.MagicMock(spec=AxisAlignedBoundingBox)
    # Simulate having a non-empty element.
    inst._classification = mock.MagicMock(spec_set=ClassificationElement)
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
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    celem = mock.MagicMock(spec_set=ClassificationElement)
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
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    celem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulate an empty ClassificationElement
    celem.has_classifications.return_value = False

    inst = MemoryDetectionElement(0)
    inst._bbox = bbox
    inst._classification = celem

    assert inst.has_detection() is False


def test_get_bbox():
    """ Test successfully getting the detection bounding box. """
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    inst = MemoryDetectionElement(0)
    inst._bbox = bbox
    assert inst.get_bbox() == bbox


def test_get_bbox_no_bbox():
    """ Test attempting to get a bbox when none is set. """
    inst = MemoryDetectionElement(0)
    with pytest.raises(NoDetectionError,
                       match="Missing detection bounding box for "
                             "in-memory detection with UUID 0"):
        inst.get_bbox()


def test_get_classification():
    """ Test successfully getting the detection classification element. """
    c_elem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulate a populated ClassificationElement
    c_elem.has_classifications.return_value = True
    inst = MemoryDetectionElement(0)
    inst._classification = c_elem
    assert inst.get_classification() == c_elem


def test_get_classification_no_classification():
    """ Test attempting to get the classification element when none is set. """
    inst = MemoryDetectionElement(0)
    with pytest.raises(NoDetectionError,
                       match="Missing or empty classification for in-memory "
                             "detection with UUID 0"):
        inst.get_classification()


def test_get_classification_empty_classification():
    """ Test attempting to get the classification element when it is empty. """
    c_elem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulate an empty ClassificationElement
    c_elem.has_classifications.return_value = False

    inst = MemoryDetectionElement(0)
    inst._classification = c_elem
    with pytest.raises(NoDetectionError,
                       match="Missing or empty classification for in-memory "
                             "detection with UUID 0"):
        inst.get_classification()


def test_get_detection():
    """ Test successfully getting the detection components. """
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    c_elem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulate a populated ClassificationElement
    c_elem.has_classifications.return_value = True

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
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    celem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulate an empty ClassificationElement
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
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    c_elem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulating that c_elem is a valid, populated classification element
    c_elem.has_classifications.return_value = True

    d_elem = MemoryDetectionElement(0)
    # noinspection PyTypeChecker
    r = d_elem.set_detection(bbox, c_elem)

    assert d_elem._bbox is bbox
    assert d_elem._classification is c_elem
    assert r is d_elem, "set_detection return was not self"


def test_set_detection_invalid_bbox():
    """
    Test that an exception is raise when a valid bounding box was
    not provided.
    """
    bbox = 'not bbox'
    c_elem = mock.MagicMock(spec_set=ClassificationElement)
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
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
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
    bbox = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    c_elem = mock.MagicMock(spec_set=ClassificationElement)
    # Simulate an empty ClassificationElement
    c_elem.has_classifications.return_value = False

    with pytest.raises(ValueError,
                       match="Given an empty ClassificationElement instance."):
        # noinspection PyTypeChecker
        MemoryDetectionElement(0).set_detection(bbox, c_elem)
