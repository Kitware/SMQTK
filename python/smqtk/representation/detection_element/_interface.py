import abc

from smqtk.exceptions import NoDetectionError
from smqtk.representation import SmqtkRepresentation
from smqtk.utils.dict import merge_dict
from smqtk.utils.plugin import Pluggable


class DetectionElement (SmqtkRepresentation, Pluggable):
    """
    Representation of a spatial detection.
    """

    __slots__ = ('_uuid',)

    @classmethod
    def get_default_config(cls):
        # Override from Configurable.
        default = super(DetectionElement, cls).get_default_config()
        # Remove runtime positional argument(s).
        del default['uuid']
        return default

    # noinspection PyMethodOverriding
    @classmethod
    def from_config(cls, config_dict, uuid, merge_default=True):
        """
        Override of
        :meth:`smqtk.utils.configuration.Configurable.from_config` with the
        added runtime argument ``uuid``. See parent method documentation for
        details.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param collections.Hashable uuid:
            UUID to assign to the produced DetectionElement.

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: DetectionElement

        """
        # Override from Configurable
        # Handle passing of runtime positional argument(s).
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)
        config_dict['uuid'] = uuid
        return super(DetectionElement, cls).from_config(config_dict,
                                                        merge_default=False)

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

    __hash__ = None

    def __eq__(self, other):
        """
        Equality of two detections is defined by their equal spatial overlap
        AND their equivalent classification.

        When one element does not contain detection information but the other
        does, the two elements are of course considered NOT equal.
        If *neither* elements contain detection information, they are defined
        as NOT equal (undefined).

        :param DetectionElement other: Other detection element.
        :return: True if the two detections are equal in spacial overlap and
            classification.
        """
        try:
            s_bb, s_ce = self.get_detection()
            o_bb, o_ce = other.get_detection()
            return s_bb == o_bb and s_ce == o_ce
        except NoDetectionError:
            return False

    def __ne__(self, other):
        return not (self == other)

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

    #
    # Abstract methods
    #

    @abc.abstractmethod
    def __getstate__(self):
        return {
            '_uuid': self._uuid,
        }

    @abc.abstractmethod
    def __setstate__(self, state):
        self._uuid = state['_uuid']

    @abc.abstractmethod
    def has_detection(self):
        """
        :return: Whether or not this container currently contains a valid
            detection bounding box and classification element (must be
            non-zero).
        :rtype: bool
        """

    @abc.abstractmethod
    def get_bbox(self):
        """
        :return: The spatial bounding box of this detection.
        :rtype: smqtk.representation.AxisAlignedBoundingBox

        :raises NoDetectionError: No detection AxisAlignedBoundingBox set yet.
        """

    @abc.abstractmethod
    def get_classification(self):
        """
        :return: The classification element of this detection.
        :rtype: smqtk.representation.ClassificationElement

        :raises NoDetectionError: No detection ClassificationElement set yet or
            the element is empty.
        """

    @abc.abstractmethod
    def get_detection(self):
        """
        :return: The paired spatial bounding box and classification element of
            this detection.
        :rtype: (smqtk.representation.AxisAlignedBoundingBox,
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
