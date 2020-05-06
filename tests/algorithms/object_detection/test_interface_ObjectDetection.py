import unittest.mock as mock

from smqtk.algorithms.object_detection import ObjectDetector
from smqtk.representation import AxisAlignedBoundingBox, DataElement


def test_gen_detection_uuid():
    """
    Test that, for the same input data, we consistently get the same UUID
    for some number of repetitions.
    """
    # Arbitrary experimental input values
    t_uuid = "SomeDataElementChecksumUUID"
    t_bbox = AxisAlignedBoundingBox([2.6, 82], [98, 83.7])
    t_labels = ('l1', 'l2', 42)
    # Number of repetitions to test over.
    n = 100

    initial_uuid = ObjectDetector._gen_detection_uuid(t_uuid, t_bbox,
                                                      t_labels)
    for i in range(n):
        ex_uuid = ObjectDetector._gen_detection_uuid(t_uuid, t_bbox,
                                                     t_labels)
        assert initial_uuid == ex_uuid, \
            "Found detection UUID ('{}', iter {}) unexpectedly deviated " \
            "from initial ('{}')" \
            .format(ex_uuid, i, initial_uuid)

    # Order of input labels should not matter.
    t_labels_otherorder = ('l2', 42, 'l1')
    assert initial_uuid == ObjectDetector._gen_detection_uuid(
        t_uuid, t_bbox, t_labels_otherorder), \
        "Rearranging label order caused UUID deviance, this shouldn't " \
        "have happened."


def test_gen_detection_uuid_perturbations():
    """
    Test that some minor variances in input data
    We obviously cannot check everything, so this is just a basic test.
    """
    # Arbitrary experimental input values
    t_uuid = "SomeDataElementChecksumUUID"
    t_bbox = AxisAlignedBoundingBox([2.6, 82], [98, 83.7])
    t_labels = ('l1', 'l2', 42)

    initial_uuid = ObjectDetector._gen_detection_uuid(t_uuid, t_bbox,
                                                      t_labels)

    # perturb t_uuid
    assert initial_uuid != ObjectDetector._gen_detection_uuid(
            t_uuid[:-1]+'.', t_bbox, t_labels), \
        "t_uuid perturbation resulted in same UUID"
    # perturb t_bbox
    assert initial_uuid != ObjectDetector._gen_detection_uuid(
            t_uuid, AxisAlignedBoundingBox([2.6, 82.1], [98, 83.7]),
            t_labels), \
        "t_bbox perturbation resulted in same UUID"
    assert initial_uuid != ObjectDetector._gen_detection_uuid(
            t_uuid, t_bbox, ('l1', 'l3', 42)), \
        "t_labels perturbation resulted in same UUID"


def test_detect_objects():
    """
    Test expected ``detect_objects`` behavior when ``_detect_objects``
    yields expected values.
    """
    # Test Inputs
    t_de1_uuid = "test uuid"
    t_de1 = mock.MagicMock(spec=DataElement)
    t_de1.uuid.return_value = t_de1_uuid
    # Expected outputs of _detect_objects
    t_det1 = (AxisAlignedBoundingBox([0, 0], [1, 1]),
              {'l1': 0, 'l2': 1})
    t_det2 = (AxisAlignedBoundingBox([1, 1], [2, 2]),
              {'l1': 1, 'l3': 0})

    # Mock instance of ObjectDetector mocking _detect_objects.
    m_inst = mock.MagicMock(spec=ObjectDetector)
    m_inst._detect_objects.return_value = (t_det1, t_det2)
    m_inst._gen_detection_uuid = \
        mock.Mock(wraps=ObjectDetector._gen_detection_uuid)

    dets_list = list(ObjectDetector.detect_objects(m_inst, t_de1))

    m_inst._detect_objects.assert_called_once_with(t_de1)
    assert m_inst._gen_detection_uuid.call_count == 2
    m_inst._gen_detection_uuid.assert_any_call(t_de1_uuid, t_det1[0],
                                               t_det1[1].keys())
    m_inst._gen_detection_uuid.assert_any_call(t_de1_uuid, t_det2[0],
                                               t_det2[1].keys())
    assert len(dets_list) == 2

    # Assert detections returned have the expected properties
    assert dets_list[0].get_detection()[0] == t_det1[0]
    assert dets_list[0].get_detection()[1].get_classification() == t_det1[1]
    assert dets_list[1].get_detection()[0] == t_det2[0]
    assert dets_list[1].get_detection()[1].get_classification() == t_det2[1]
