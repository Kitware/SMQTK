from smqtk.representation import (
    SmqtkRepresentation,
    ClassificationElement
)
from smqtk.utils.configuration import (
    cls_conf_from_config_dict,
    cls_conf_to_config_dict,
    make_default_config,
)
from smqtk.utils.dict import merge_dict


__author__ = "paul.tunison@kitware.com"


class ClassificationElementFactory (SmqtkRepresentation):
    """
    Factory class for producing ClassificationElement instances of a specified
    type and configuration.
    """

    # noinspection PyShadowingBuiltins
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
        return make_default_config(ClassificationElement.get_impls())

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
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        ce_type, ce_conf = cls_conf_from_config_dict(
            config_dict,  ClassificationElement.get_impls()
        )
        return ClassificationElementFactory(ce_type, ce_conf)

    def get_config(self):
        return cls_conf_to_config_dict(self.type, self.type_config)

    # noinspection PyShadowingBuiltins
    def new_classification(self, type, uuid):
        """
        Create a new ClassificationElement instance of the configured
        implementation.

        :param type: Type of classifier. This is usually the name of the
            classifier that generated this result.
        :type type: str

        :param uuid: UUID to associate with the classification.
        :type uuid: collections.abc.Hashable

        :return: New ClassificationElement instance.
        :rtype: smqtk.representation.ClassificationElement

        """
        return self.type.from_config(self.type_config, type, uuid)

    # noinspection PyShadowingBuiltins
    def __call__(self, type, uuid):
        """
        Create a new ClassificationElement instance of the configured
        implementation.

        :param type: Type of classifier. This is usually the name of the
            classifier that generated this result.
        :type type: str

        :param uuid: UUID to associate with the classification.
        :type uuid: collections.abc.Hashable

        :return: New ClassificationElement instance.
        :rtype: smqtk.representation.ClassificationElement

        """
        return self.new_classification(type, uuid)
