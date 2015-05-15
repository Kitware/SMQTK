__author__ = 'purg'

import abc
import logging


class DataElement (object):
    """
    Base abstract class for a data element.

    Basic data elements have a UUID, some byte content, and a content type.

    DataElement implementations should be picklable for serialization, as some
    DataSet implementations will desire such functionality.

    """
    __metaclass__ = abc.ABCMeta

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    @abc.abstractmethod
    def __hash__(self):
        return

    @abc.abstractmethod
    def content_type(self):
        """
        :return: Standard type/subtype string for this data
            element, or None if the content type is unknown.
        :rtype: str or None
        """
        return

    @abc.abstractmethod
    def md5(self):
        """
        :return: MD5 hex string of the data content.
        :rtype: str
        """
        return

    @abc.abstractmethod
    def uuid(self):
        """
        UUID for this data element. This many take different forms from integers
        to strings to a uuid.UUID instance. This must return a hashable data
        type.

        :return: UUID value for this data element. This return value should be
            hashable.
        :rtype: collections.Hashable

        """
        return

    @abc.abstractmethod
    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return

    read = get_bytes
