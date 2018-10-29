from smqtk.utils import Configurable, plugin, SmqtkObject


# noinspection PyAbstractClass
class SmqtkAlgorithm (SmqtkObject, Configurable, plugin.Pluggable):
    """
    Parent class for all algorithm interfaces.
    """

    @property
    def name(self):
        """
        :return: The name of this class type.
        :rtype: str
        """
        return self.__class__.__name__
