import abc

from smqtk.utils.plugin import Pluggable


class DummyInterface (Pluggable):

    @abc.abstractmethod
    def inst_method(self, val):
        """
        dummy abstract function
        """
