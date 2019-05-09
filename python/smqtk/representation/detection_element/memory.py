from smqtk.exceptions import NoDetectionError
from smqtk.representation import (
    AxisAlignedBoundingBox,
    ClassificationElement,
    DetectionElement
)


class MemoryDetectionElement (DetectionElement):  # lgtm[py/missing-equals]
    """
    In-memory backend of the DetectionElement representation interface.  This
    implementation has no persistence.

    See ``DetectionElement`` for documentation on abstract method implemented
    here.
    """

    __slots__ = ('_bbox', '_classification')

    @classmethod
    def is_usable(cls):
        # In-memory implementation does not require any additional
        # dependencies.
        return True

    def __init__(self, uuid):
        super(MemoryDetectionElement, self).__init__(uuid)
        #: :type: None | AxisAlignedBoundingBox
        self._bbox = None
        #: :type: None | ClassificationElement
        self._classification = None

    def __getstate__(self):
        return {
            'parent': super(MemoryDetectionElement, self).__getstate__(),
            'bbox': self._bbox,
            'classification': self._classification,
        }

    def __setstate__(self, state):
        super(MemoryDetectionElement, self).__setstate__(state['parent'])
        self._bbox = state['bbox']
        self._classification = state['classification']

    def get_config(self):
        # No additional constructor parameters for in-memory implementation.
        return {}

    def has_detection(self):
        # We are a valid detection if our components are non-null and
        # True-evaluation (meaning they have valid contents).
        return None not in (self._bbox, self._classification) \
               and self._classification.has_classifications()

    def get_bbox(self):
        if not self._bbox:
            raise NoDetectionError("Missing detection bounding box for "
                                   "in-memory detection with UUID {}"
                                   .format(self.uuid))
        return self._bbox

    def get_classification(self):
        if not (self._classification and
                self._classification.has_classifications()):
            raise NoDetectionError("Missing or empty classification for "
                                   "in-memory detection with UUID {}"
                                   .format(self.uuid))
        return self._classification

    def get_detection(self):
        if not (self._bbox and self._classification.has_classifications()):
            raise NoDetectionError("Missing detection bounding box or "
                                   "missing/invalid classification for "
                                   "in-memory detection with UUID {}"
                                   .format(self.uuid))
        return self._bbox, self._classification

    def set_detection(self, bbox, classification_element):
        if not isinstance(bbox, AxisAlignedBoundingBox):
            raise ValueError("Provided an invalid AxisAlignedBoundingBox "
                             "instance. Given '{}' (type={})."
                             .format(bbox, type(bbox).__name__))
        if not isinstance(classification_element, ClassificationElement):
            raise ValueError("Provided an invalid ClassificationElement "
                             "instance. Given '{}' (type={})."
                             .format(classification_element,
                                     type(classification_element).__name__))
        if not classification_element.has_classifications():
            raise ValueError("Given an empty ClassificationElement instance.")
        self._bbox = bbox
        self._classification = classification_element
        return self
