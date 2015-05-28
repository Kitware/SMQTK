__author__ = 'purg'

import abc
import logging
import numpy


class DescriptorElement (object):
    """
    Abstract descriptor vector container. The intent of this structure is to
    hide the specific method of storage of data (e.g. memory, file, database,
    etc.).
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, source_descriptor):
        """
        :param source_descriptor: ContentDescriptor that generated this
            descriptor.
        :type source_descriptor: smqtk.content_description.ContentDescriptor
        """
        self._source_descriptor_label = source_descriptor.__class__.__name__

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __hash__(self):
        return hash(self.uuid())

    def __eq__(self, other):
        if isinstance(other, DescriptorElement):
            b = self.vector() == other.vector()
            if isinstance(b, numpy.core.multiarray.ndarray):
                return b.all()
            else:
                return b
        return False

    def __ne__(self, other):
        return not (self == other)

    def type(self):
        """
        :return: Type label type of the ContentDescriptor that generated this
            vector.
        :rtype: str
        """
        return self._source_descriptor_label

    ###
    # Abstract methods
    #

    @abc.abstractmethod
    def uuid(self):
        """
        :return: Unique ID for this vector.
        :rtype: collections.Hashable
        """
        return

    @abc.abstractmethod
    def vector(self):
        """
        :return: The descriptor vector as a numpy array.
        :rtype: numpy.core.multiarray.ndarray
        """
        return
