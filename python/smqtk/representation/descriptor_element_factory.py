from smqtk.representation import \
    SmqtkRepresentation, \
    DescriptorElement
from smqtk.utils.configuration import (
    cls_conf_from_config_dict,
    cls_conf_to_config_dict,
    make_default_config,
)
from smqtk.utils.dict import merge_dict


__author__ = "paul.tunison@kitware.com"


class DescriptorElementFactory (SmqtkRepresentation):
    """
    Factory class for producing DescriptorElement instances of a specified type
    and configuration.
    """

    def __init__(self, d_type, type_config):
        """
        Initialize the factory to produce DescriptorElement instances of the
        given type and configuration.

        :param d_type: Type of descriptor element this factory should produce.
        :type d_type: type

        :param type_config: Initialization parameter dictionary that should
            contain all additional construction parameters for the provided type
            except for the expected `type_str` and `uuid` arguments that should
            be the first and second positional arguments respectively.
        :type type_config: dict

        """
        #: :type: type | smqtk.representation.DescriptorElement
        self._d_type = d_type
        self._d_type_config = type_config

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        return make_default_config(DescriptorElement.get_impls())

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: DescriptorElementFactory

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        de_type, de_conf = cls_conf_from_config_dict(
            config_dict, DescriptorElement.get_impls()
        )
        return DescriptorElementFactory(de_type, de_conf)

    def get_config(self):
        return cls_conf_to_config_dict(self._d_type, self._d_type_config)

    def new_descriptor(self, type_str, uuid):
        """
        Create a new DescriptorElement instance of the configured implementation

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: UUID to associate with the descriptor
        :type uuid: collections.Hashable

        :return: New DescriptorElement instance
        :rtype: smqtk.representation.DescriptorElement

        """
        return self._d_type.from_config(self._d_type_config, type_str, uuid)

    def __call__(self, type_str, uuid):
        """
        Create a new DescriptorElement instance of the configured implementation

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: UUID to associate with the descriptor
        :type uuid: collections.Hashable

        :return: New DescriptorElement instance
        :rtype: smqtk.representation.DescriptorElement

        """
        return self.new_descriptor(type_str, uuid)
