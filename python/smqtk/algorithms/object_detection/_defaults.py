"""
Default values and instances for the ObjectDetection interface.
"""
# - Using the same default factory for ObjectDetector as the Classifier
#   interface.
# - Providing classifier default here for convenience.
# noinspection PyProtectedMember
from smqtk.algorithms.classifier._defaults import DFLT_CLASSIFIER_FACTORY  # lgtm[py/unused-import]
from smqtk.representation import DetectionElementFactory
from smqtk.representation.detection_element.memory \
    import MemoryDetectionElement

DFLT_DETECTION_FACTORY = DetectionElementFactory(
    MemoryDetectionElement, {}
)
