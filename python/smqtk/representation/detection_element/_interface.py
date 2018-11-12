import abc

from smqtk.representation import SmqtkRepresentation
from smqtk.utils.plugin import Pluggable


class DetectionElement (SmqtkRepresentation, Pluggable):
    """
    Representation of a spatial detection.
    """

    __slots__ = ('_uuid',)

    def __init__(self, uuid):
        """
        Initialize a new detection element with the given ``uuid``.

        All DetectionElement classes will take a ``uuid`` parameter as the
        first positional argument. This parameter is not configurable and is
        only specified at runtime. Implementing classes should not include
        ``uuid`` in ``get_config`` returns.

        :param collections.Hashable uuid:
            Unique ID reference of the detection.

        """
        super(DetectionElement, self).__init__()
        self._uuid = uuid

    def __hash__(self):
        return hash(self._uuid)

    def __repr__(self):
        # using "{{...}}" to skip .format activation.
        return "{:s}{{uuid: {}}}".format(self.__class__.__name__, self._uuid)

    def __nonzero__(self):
        """
        A DetectionElement is considered non-zero if ``has_detection`` returns
        True. See method documentation for details.

        :return: True if this instance is non-zero (see above), false
            otherwise.
        :rtype: bool
        """
        return self.has_detection()

    __bool__ = __nonzero__

    @property
    def uuid(self):
        return self._uuid

    @abc.abstractmethod
    def has_detection(self):
        """
        :return: Whether or not this container currently contains a valid
            detection bounding box and classification element (must be
            non-zero).
        :rtype: bool
        """

    @abc.abstractmethod
    def get_detection(self):
        """
        :return: The paired spatial bounding box and classification element of
            this detection.
        :rtype: (smqtk.representation.BoundingBox,
                 smqtk.representation.ClassificationElement)

        :raises NoDetectionError: No detection AxisAlignedBoundingBox and
            ClassificationElement set yet.

        """

    @abc.abstractmethod
    def set_detection(self, bbox, classification_element):
        """
        Set a bounding box and classification element to this detection
        element.

        :param smqtk.representation.AxisAlignedBoundingBox bbox:
            Spatial bounding box instance.

        :param smqtk.representation.ClassificationElement classification_element:
            The classification of this detection.

        :raises ValueError: No, or invalid, AxisAlignedBoundingBox or
            ClassificationElement was provided.

        :returns: Self
        :rtype: DetectionElement

        """
