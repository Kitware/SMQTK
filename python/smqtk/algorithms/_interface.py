from smqtk.utils import SmqtkObject
from smqtk.utils.configuration import Configurable
from smqtk.utils.plugin import Pluggable


# noinspection PyAbstractClass
class SmqtkAlgorithm (SmqtkObject, Configurable, Pluggable):
    """
    Parent class for all algorithm interfaces.
    """

    __slots__ = ()

    @property
    def name(self):
        """
        :return: The name of this class type.
        :rtype: str
        """
        return self.__class__.__name__
