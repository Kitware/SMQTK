"""
Default values and instances for the ObjectDetection interface.
"""
# noinspection PyProtectedMember,PyUnresolvedReferences
# - Using the same default factory for ObjectDetector as the Classifier
#   interface.
from smqtk.algorithms.classifier._defaults import DFLT_CLASSIFIER_FACTORY
from smqtk.representation import DetectionElementFactory
from smqtk.representation.detection_element.memory \
    import MemoryDetectionElement

DFLT_DETECTION_FACTORY = DetectionElementFactory(
    MemoryDetectionElement, {}
)
