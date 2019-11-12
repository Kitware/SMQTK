import logging


class SmqtkObject (object):
    """
    Highest level object interface for classes defined in SMQTK.

    Currently defines logging methods.

    """

    __slots__ = ()

    @classmethod
    def get_logger(cls):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((cls.__module__, cls.__name__)))

    @property
    def _log(self):
        """
        :return: logging object for this class as a property
        :rtype: logging.Logger
        """
        return self.get_logger()
