from smqtk.representation import (
    DetectionElement,
    SmqtkRepresentation,
)
from smqtk.utils.configuration import (
    cls_conf_from_config_dict,
    cls_conf_to_config_dict,
    make_default_config,
)
from smqtk.utils.dict import merge_dict


class DetectionElementFactory (SmqtkRepresentation):
    """
    Factory class for producing DetectionElement instances of a specified type
    and configuration.
    """

    __slots__ = ('_elem_type', '_elem_config')

    @classmethod
    def get_default_config(cls):
        # Override from Configurable
        return make_default_config(DetectionElement.get_impls())

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        # Override from Configurable
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        elem_type, elem_conf = cls_conf_from_config_dict(
            config_dict, DetectionElement.get_impls()
        )
        return DetectionElementFactory(elem_type, elem_conf)

    def __init__(self, elem_type, elem_config):
        """
        Initialize the factory to produce DetectionElement instance of the
        given type and configuration.

        :param type[DetectionElement] elem_type:
            Instantiable type of DetectionElement this factory should produce.
        :param dict elem_config:
            JSON-compliant dictionary configuration that is to be supplied to
            the input type's ``from_config`` class method. If the ``uuid`` key
            happens to be present in this dictionary it will be ignored
        """
        self._elem_type = elem_type
        self._elem_config = elem_config

    def get_config(self):
        return cls_conf_to_config_dict(self._elem_type, self._elem_config)

    def new_detection(self, uuid):
        """
        Create a new DetectionElement instance o the configured implementation.

        :param collections.Hashable uuid:
            UUID to assign the element.

        :return: New DetectionElement instance.
        :rtype: DetectionElement

        """
        # noinspection PyUnresolvedReferences
        return self._elem_type.from_config(self._elem_config, uuid)

    __call__ = new_detection
