__author__ = 'purg'

import abc


class DataElement (object):
    """
    Base abstract class for a data element.

    Basic data elements have a UUID, some byte content, and a content type.

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __hash__(self):
        return

    @abc.abstractproperty
    def uuid(self):
        """
        UUID for this data element. This many take different forms from integers
        to strings to a uuid.UUID instance.

        :return: UUID value for this data element. This return value should be
            hashable.

        """
        return

    @abc.abstractproperty
    def content_type(self):
        """
        :return: Standard type/subtype string for this data
            element, or None if the content type is unknown.
        :rtype: str or None
        """
        return

    @abc.abstractmethod
    def get_bytes(self):
        """
        :return: Get the byte stream for this data element.
        :rtype: bytes
        """
        return
