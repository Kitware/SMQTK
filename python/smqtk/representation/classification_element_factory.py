from smqtk.representation import (
    SmqtkRepresentation,
    get_classification_element_impls
)
from smqtk.utils.plugin import make_config
from smqtk.utils import merge_dict


__author__ = "paul.tunison@kitware.com"


class ClassificationElementFactory (SmqtkRepresentation):
    """
    Factory class for producing ClassificationElement instances of a specified
    type and configuration.
    """

    def __init__(self, type, type_config):
        """
        Initialize the factory to produce ClassificationElement instances of the
        given type from the given configuration.

        :param type: Python implementation type of the ClassifierElement to
            produce
        :type type: type

        :param type_config: Configuration dictionary that will be passed
            ``from_config`` class method of given ``type``.
        :type type_config: dict

        """
        #: :type: type | smqtk.representation.ClassificationElement
        self.type = type
        self.type_config = type_config

    #
    # Class methods
    #

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
        return make_config(get_classification_element_impls())

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
        :rtype: ClassificationElementFactory

        """
        if merge_default:
            mc = cls.get_default_config()
            merge_dict(mc, config_dict)
            config_dict = mc

        return ClassificationElementFactory(
            get_classification_element_impls()[config_dict['type']],
            config_dict[config_dict['type']]
        )

    def get_config(self):
        type_name = self.type.__name__
        return {
            "type": type_name,
            type_name: self.type_config,
        }

    def new_classification(self, type, uuid):
        """
        Create a new ClassificationElement instance of the configured
        implementation.

        :param type: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type: str

        :param uuid: UUID to associate with the descriptor
        :type uuid: collections.Hashable

        :return: New ClassificationElement instance
        :rtype: smqtk.representation.ClassificationElement

        """
        return self.type.from_config(self.type_config, type, uuid)

    def __call__(self, type, uuid):
        """
        Create a new ClassificationElement instance of the configured
        implementation.

        :param type: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type: str

        :param uuid: UUID to associate with the descriptor
        :type uuid: collections.Hashable

        :return: New ClassificationElement instance
        :rtype: smqtk.representation.ClassificationElement

        """
        return self.new_classification(type, uuid)
