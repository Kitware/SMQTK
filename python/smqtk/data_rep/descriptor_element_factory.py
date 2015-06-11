__author__ = 'purg'


class DescriptorElementFactory (object):
    """
    Factory class for DescriptorElement instances of a specific type and
    configuration.
    """

    def __init__(self, d_type, init_params):
        """
        :param d_type: Type of descriptor element this factory should produce.
        :type d_type: type
        :param init_params: Initialization parameter dictionary that should
            contain all
        :type init_params: dict
        """
        self._d_type = d_type
        self._init_params = init_params

    def new_descriptor(self, type_str, uuid):
        return self._d_type(type_str, uuid, **self._init_params)

    def __call__(self, type_str, uuid):
        return self.new_descriptor(type_str, uuid)
