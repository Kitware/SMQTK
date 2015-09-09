from smqtk.data_rep import get_descriptor_element_impls
from smqtk.utils.configurable_interface import Configurable
from smqtk.utils.plugin import make_config


__author__ = 'purg'


class DescriptorElementFactory (Configurable):
    """
    Factory class for DescriptorElement instances of a specific type and
    configuration.
    """

    @classmethod
    def default_config(cls):
        return make_config(get_descriptor_element_impls)

    @classmethod
    def from_config(cls, config_dict):
        return DescriptorElementFactory(
            get_descriptor_element_impls()[config_dict['type']],
            config_dict[config_dict['type']]
        )

    def __init__(self, d_type, init_params):
        """
        :param d_type: Type of descriptor element this factory should produce.
        :type d_type: type
        :param init_params: Initialization parameter dictionary that should
            contain all additional construction parameters for the provided type
            except for the expected `type_str` and `uuid` arguments that should
            be the first and second positional arguments respectively.
        :type init_params: dict
        """
        self._d_type = d_type
        self._init_params = init_params

    def get_config(self):
        d_type_name = self._d_type.__class__.__name__
        return {
            'type': d_type_name,
            d_type_name: self._init_params,
        }

    def new_descriptor(self, type_str, uuid):
        """
        Create a new DescriptorElement instance of the configured implementation

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: UUID to associate with the descriptor
        :type uuid: collections.Hashable

        :return: New DescriptorElement instance
        :rtype: smqtk.data_rep.DescriptorElement

        """
        return self._d_type(type_str, uuid, **self._init_params)

    def __call__(self, type_str, uuid):
        """
        Create a new DescriptorElement instance of the configured implementation

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: UUID to associate with the descriptor
        :type uuid: collections.Hashable

        :return: New DescriptorElement instance
        :rtype: smqtk.data_rep.DescriptorElement

        """
        return self.new_descriptor(type_str, uuid)
